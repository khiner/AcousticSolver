/** (c) 2024 Kangrui Xue
 *
 * \file ModalSound.cpp
 * \brief Implements Solver class
 * 
 * Based on code by Jui-Hsien Wang (https://github.com/jhwang7628/openpbso)
 * 
 * TODO: in general, this library is still pretty barebones (and in need of cleanup)
 */

#include "ModalSound.h"


namespace ModalSound {

Solver::Solver(const std::string &dataPrefix, const std::string &meshFile, const std::string &matName, double dt)
{
    _dt = dt;
    _dataPrefix = dataPrefix; _matName = matName;

    // First, read the modes into ModeData struct
    ModeData modeData; 
    std::string modeFile = dataPrefix + ".modes";
    modeData.read(modeFile.c_str());

    const int N_modes = modeData.numModes();
    const int N_DOF = modeData.numDOF();
    
    // Copy data into Eigen matrices / vectors
    _eigenValues.resize(N_modes);
    _eigenVectors.resize(N_DOF, N_modes);
    for (int m_idx = 0; m_idx < N_modes; ++m_idx)
    {
        _eigenValues(m_idx) = modeData.omegaSquared(m_idx);
        for (int d_idx = 0; d_idx < N_DOF; ++d_idx)
        {
            _eigenVectors(d_idx, m_idx) = modeData.mode(m_idx).at(d_idx);
        }
    }
    CullNonSurfaceModes();

    // Read mesh and precompute curvature + inertia
    bool success = igl::read_triangle_mesh(meshFile, _V, _F);
    
    Eigen::MatrixXd PD1, PD2; Eigen::VectorXd PV1, PV2;
    igl::principal_curvature(_V, _F, PD1, PD2, PV1, PV2);
    _curvature = (PV1 + PV2) / 2.;  // mean curvature

    ComputeInertia();

    // Load acceleration noise impulses
    LoadImpulses();

    // Initialize modal dynamics vectors
    _q_p.setZero(N_modes);
    _q_c.setZero(N_modes);
    _q_n.setZero(N_modes);
    _q_nn.setZero(N_modes);

    _qDot_c_plus.setZero(N_modes);
    _qDDot_c.setZero(N_modes);
}


struct ModalMaterial  // Default ABS Plastic
{
    int id = 0;
    double alpha = 30.;
    double beta = 1e-6;
    double density = 1070.0; // assuming we always have constant, uniform density
    double youngsModulus = 1.4e9;
    double poissonRatio = 0.35;

    double one_minus_nu2_over_E() { return (1.0 - std::pow(poissonRatio, 2)) / youngsModulus; }
    double xi(const double& omega_i) { return 0.5 * (alpha / omega_i + beta * omega_i); } // eq. 10, xi = [0, 1]
    double omega_di(const double& omega_i) { return omega_i * sqrt(1.0 - pow(xi(omega_i), 2)); } // eq. 12.
};


double Solver::step(double time)
{
    std::vector<ImpactRecord> activeImpacts;
    Eigen::VectorXd forceTimestep;

    _impulseSeries.GetForces(time, activeImpacts);

    // Get force in modal space
    forceTimestep.setZero(_eigenVectors.cols());
    const int N_records = activeImpacts.size();

    // For each impact within this timestep, add forces to the corresponding modes
    for (int rec_idx = 0; rec_idx < N_records; ++rec_idx)
    {
        const Eigen::Vector3d force = activeImpacts[rec_idx].impactVector;
        const int& vertexID = activeImpacts[rec_idx].appliedVertex;

        forceTimestep += _eigenVectors.row(3*vertexID + 0) * force[0]
            + _eigenVectors.row(3*vertexID + 1) * force[1]
            + _eigenVectors.row(3*vertexID + 2) * force[2];
    }

    // TODO: support other materials
    ModalMaterial material;
    if (_matName == "Glass")
    {
        material.id = 1; material.alpha = 1.; material.beta = 1e-7;
        material.density = 2600.0; material.youngsModulus = 6.2e10; material.poissonRatio = 0.2;
    }

    // Compute filter coefficients for time step
    const int N_modes = _eigenVectors.cols();
    Eigen::VectorXd _coeff_qNew = Eigen::VectorXd::Zero(N_modes);
    Eigen::VectorXd _coeff_qOld = Eigen::VectorXd::Zero(N_modes);
    Eigen::VectorXd _coeff_Q = Eigen::VectorXd::Zero(N_modes);
    for (int i = 0; i < N_modes; i++)
    {
        double omega = std::sqrt(_eigenValues[i] / material.density);
        double omega_di = material.omega_di(omega);
        
        double xi = material.xi(omega);
        double _epsilon = std::exp(-xi * omega * _dt);
        double _theta = omega_di * _dt;
        double _gamma = std::asin(xi);

        _coeff_qNew[i] = 2. * _epsilon * std::cos(_theta);
        _coeff_qOld[i] = _epsilon * _epsilon;
        _coeff_Q[i] = 2. / (3. * omega * omega_di)
            * (_epsilon * std::cos(_theta + _gamma) - _epsilon * _epsilon * std::cos(2. * _theta + _gamma))
            / material.density;
    }

    // Step the system
    _q_nn = _coeff_qNew.cwiseProduct(_q_n) - _coeff_qOld.cwiseProduct(_q_c) + _coeff_Q.cwiseProduct(forceTimestep);
   
    _qDot_c_plus = (_q_n - _q_c) / _dt;
    _qDDot_c = (_q_n + _q_p - 2.0 * _q_c) / (_dt * _dt);

    // Update pointers
    _q_p = _q_c;
    _q_c = _q_n;
    _q_n = _q_nn;

    return _qDDot_c.sum();
}


void Solver::CullNonSurfaceModes()
{
    std::ifstream geoFile(_dataPrefix + ".geo.txt");
    int mapSize; geoFile >> mapSize;

    int tet_idx, surf_idx; double nx, ny, nz, area;
    std::map<int, int> idxMap;
    std::vector<Eigen::Vector3d> normals;
    while (geoFile >> tet_idx >> surf_idx >> nx >> ny >> nz >> area)
    {
        idxMap.insert(std::pair<int, int>(tet_idx, surf_idx));
        normals.push_back(Eigen::Vector3d({nx, ny, nz}));
    }

    const int N_surfaceVertices = idxMap.size();
    const int N_volumeVertices = _eigenVectors.rows() / 3;
    const int N_modes = _eigenVectors.cols();

    Eigen::MatrixXd culledEigenVectors(N_surfaceVertices * 3, N_modes);
    _eigenVectorsNormal.resize(N_surfaceVertices, N_modes);
    _normals.resize(N_surfaceVertices, 3);
    for (int vol_idx = 0; vol_idx < N_volumeVertices; ++vol_idx)
    {
        if (idxMap.count(vol_idx) == 0) { continue; }

        const int surf_idx = idxMap[vol_idx];
        _normals.row(surf_idx) = normals[surf_idx];

        culledEigenVectors.row(surf_idx * 3 + 0) = _eigenVectors.row(vol_idx * 3 + 0);
        culledEigenVectors.row(surf_idx * 3 + 1) = _eigenVectors.row(vol_idx * 3 + 1);
        culledEigenVectors.row(surf_idx * 3 + 2) = _eigenVectors.row(vol_idx * 3 + 2);
        _eigenVectorsNormal.row(surf_idx) = _eigenVectors.row(vol_idx * 3 + 0) * normals[surf_idx][0]
            + _eigenVectors.row(vol_idx * 3 + 1) * normals[surf_idx][1]
            + _eigenVectors.row(vol_idx * 3 + 2) * normals[surf_idx][2];
    }

    _eigenVectors = culledEigenVectors;
}


double Solver::EffectiveMass(const Eigen::Vector3d& x, const Eigen::Vector3d& n)
{
    const Eigen::Vector3d r = x - _centerOfMass;  // mass center = volume center
    const Eigen::Vector3d r_cross_n = r.cross(n.normalized());
    const Eigen::Vector3d premult = _I_Inv * r_cross_n;
    double one_over_m_corr = r_cross_n.dot(premult);
    return 1. / (1. / _mass + one_over_m_corr);
}


double Solver::EstimateContactTimeScale(int vertex_a, double contactSpeed, const Eigen::Vector3d &impulse_a)
{
    ModalMaterial material_a;
    const Eigen::Vector3d x_a = _V.row(vertex_a);
    const double m = EffectiveMass(x_a, impulse_a);
    double one_over_r = _curvature(vertex_a);
    const double one_over_E = material_a.one_minus_nu2_over_E();
    
    return 2.87 * std::pow(std::pow(m * one_over_E, 2) * std::fabs(one_over_r / contactSpeed), 0.2);
}


void Solver::LoadImpulses()
{
    std::ifstream inFile(_dataPrefix + ".impulses.txt");
    
    int objectID = -1, objectID_old = -1, count = 0; 
    char pairOrder, impulseType; 
    ImpactRecord buffer, buffer_old; 
    while (inFile >> buffer.timestamp >> objectID >> buffer.appliedVertex >> buffer.contactSpeed
                  >> buffer.impactVector[0] >> buffer.impactVector[1] >> buffer.impactVector[2]
                  >> pairOrder >> impulseType)
    {
        if (objectID == 0) 
        {
            if (impulseType == 'C')
            {
                buffer.supportLength = EstimateContactTimeScale(buffer.appliedVertex, buffer.contactSpeed, buffer.impactVector);  // TODO: factor out
                buffer.impactPosition = _V.row(buffer.appliedVertex);
                buffer.gamma = M_PI * buffer.impactVector.norm() / 2. / buffer.supportLength;
                _impulseSeries.AddImpulse(buffer);  // assumes point impulse
            }
            else { throw std::runtime_error("Unsupported impulse type: " + impulseType); }
        }
        buffer_old = buffer; 
        objectID_old = objectID; 
        count++;
    }
    _impulseSeries.Filter(); 
}


/** */
void Solver::ComputeInertia()
{
    // Read tetrahedral mesh
    std::ifstream inFile(_dataPrefix + ".tet", std::ios::in | std::ios::binary);

    int Nverts, Ntets;
    Eigen::MatrixXd V;
    Eigen::MatrixXi T;

    Eigen::Vector3d Vbuf; Eigen::Vector4i Tbuf;
    inFile.read((char *) &Nverts, sizeof(int));  // first 4 bytes empty -- skip ahead
    inFile.read((char *) &Nverts, sizeof(int));
    V.resize(Nverts, 3);
    for (int i = 0; i < Nverts; i++)
    {
        inFile.read((char *) Vbuf.data(), 3 * sizeof(double));
        V.row(i) = Vbuf;
    }
    inFile.read((char *) &Ntets, sizeof(int));
    T.resize(Ntets, 4);
    for (int i = 0; i < Ntets; i++)
    {
        inFile.read((char *) Tbuf.data(), 4 * sizeof(int));
        T.row(i) = Tbuf;
    }

    Eigen::VectorXd volumes = Eigen::VectorXd::Zero(Nverts);  // volume (scale by density to get mass)
    for (int i = 0; i < Ntets; i++)
    {
        Eigen::Vector3d p1 = V.row(T(i, 0));
        Eigen::Vector3d p2 = V.row(T(i, 1));
        Eigen::Vector3d p3 = V.row(T(i, 2));
        Eigen::Vector3d p4 = V.row(T(i, 3));

        Eigen::Vector3d ad = p2 - p1;
        Eigen::Vector3d bd = p3 - p1;
        Eigen::Vector3d cd = p4 - p1;

        double vol = (std::abs(ad.dot(bd.cross(cd))) / 6.) * 0.25;

        volumes[T(i, 0)] += vol;
        volumes[T(i, 1)] += vol;
        volumes[T(i, 2)] += vol;
        volumes[T(i, 3)] += vol;
    }
    
    double totalVolume = 0.;
    _centerOfMass = Eigen::Vector3d::Zero();
    for (int i = 0; i < Nverts; i++)
    {
        totalVolume += volumes[i];
        _centerOfMass += volumes[i] * V.row(i);
    }
    _centerOfMass /= totalVolume; // mass center

    _Inertia = Eigen::Matrix3d::Zero();
    for (int i = 0; i < Nverts; i++)
    {
        Eigen::Vector3d ri = V.row(i); ri -= _centerOfMass;
        Eigen::Matrix3d Tmp; 
        Tmp << volumes[i] * (ri[1] * ri[1] + ri[2] * ri[2]), -volumes[i] * ri[0] * ri[1], -volumes[i] * ri[0] * ri[2],
            -volumes[i] * ri[1] * ri[0], volumes[i] * (ri[0] * ri[0] + ri[2] * ri[2]), -volumes[i] * ri[1] * ri[2],
            -volumes[i] * ri[2] * ri[0], -volumes[i] * ri[2] * ri[1], volumes[i] * (ri[0] * ri[0] + ri[1] * ri[1]);
        _Inertia += Tmp;
    }

    ModalMaterial material_a;
    _I_Inv = (_Inertia * material_a.density).inverse();
    _mass = totalVolume * material_a.density;
}

} // namespace ModalSound