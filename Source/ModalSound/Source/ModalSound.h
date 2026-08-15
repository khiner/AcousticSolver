/** (c) 2024 Kangrui Xue
 * 
 * \file ModalSound.h
 * \brief Public interface for ModalSound; declares ImpactRecord, ImpulseSeries, and Solver classes
 * 
 * Based on code by Jui-Hsien Wang (https://github.com/jhwang7628/openpbso)
 * 
 * TODO: in general, this library is still pretty barebones (and in need of cleanup)
 */

#ifndef MODAL_SOUND_H
#define MODAL_SOUND_H

#include <map>

#include "igl/read_triangle_mesh.h"
#include "igl/principal_curvature.h"

#include "ModeData.h"


namespace ModalSound {

/**
 * \struct ImpactRecord
 * \brief Stores impact data needed for modal force and acceleration noise calculation
 */
struct ImpactRecord
{
    Eigen::Vector3d impactVector;
    Eigen::Vector3d impactPosition;     // in object space
    double timestamp;
    double supportLength = 0.01;        // tau
    double contactSpeed = 0.;
    double gamma = 1.;                  // TODO: better default, gamma = pi * norm(J) / (2 tau)
    int appliedVertex;
};


/**
 * \struct ImpulseSeries
 * \brief Stores a series of ImpactRecords, with functionality for filtering by time
 */
class ImpulseSeries
{
public:
    // Vertex index should be for surface triangle mesh (not volumetric tetrahedral mesh) 
    void AddImpulse(const ImpactRecord& record)
    {
        _lastImpulseTime = record.timestamp;
        _firstImpulseTime = std::min<double>(_firstImpulseTime, record.timestamp);
        _impulses.push_back(record);
    }

    void GetForces(const double& timeStart, std::vector<ImpactRecord>& records)
    {
        if (timeStart > _lastImpulseTime && timeStart < _firstImpulseTime) { return; }   // TODO: refactor (this appears in ModalShader.cu)

        records.clear();
        const int N_impulses = _impulses.size();
        for (int frame_idx = 0; frame_idx < N_impulses; ++frame_idx)
        {
            const ImpactRecord& record = _impulses.at(frame_idx);
            if (timeStart >= record.timestamp && timeStart < record.timestamp + record.supportLength)
                records.push_back(record);
        }

        const int N = records.size();
        for (int frame_idx = 0; frame_idx < N; ++frame_idx)
        {
            ImpactRecord& r = records[frame_idx];
            const double j = r.impactVector.norm();
            r.impactVector = (r.impactVector / j) * r.gamma; 
            
            double S = 0.;  // TODO: refactor (this appears in ModalShader.cu)
            if (timeStart <= r.timestamp + r.supportLength && timeStart >= r.timestamp)
                S = std::sin(M_PI * (timeStart - r.timestamp) / r.supportLength);
            r.impactVector *= S;
        }
    }

    void Filter()
    {
        const double IMPULSE_VEL_THRESHOLD = 0.05;
        for (std::vector<ImpactRecord>::iterator it = _impulses.begin(); it < _impulses.end();)
        {
            if (abs(it->contactSpeed) < IMPULSE_VEL_THRESHOLD) { it = _impulses.erase(it); }
            else { ++it; }
        }

        // update cached fields
        if (_impulses.size() > 0)
        {
            _firstImpulseTime = _impulses.at(0).timestamp;
            _lastImpulseTime = (_impulses.size() == 1) ? _firstImpulseTime + 1e-8 :
                _impulses.at(_impulses.size() - 1).timestamp;
        }
        else
        {
            _firstImpulseTime = 0.0;
            _lastImpulseTime = 0.0;
        }
    }
    double firstImpulseTime() const { return _firstImpulseTime; }
    double lastImpulseTime() const { return _lastImpulseTime; }

    std::vector<ImpactRecord> _impulses;

protected:
    double _firstImpulseTime;
    double _lastImpulseTime;
};


/**
 * \class Solver
 * \brief
 */
class Solver
{
public:
    /** Constructor */
    Solver(const std::string& dataPrefix, const std::string& meshFile, const std::string& matName, double dt);

    /** Timesteps modal oscillators */
    double step(double time);

    //void loadState(const std::string &stateFile);
    //void saveState(const std::string &stateFile);

    // Make these public for now
    Eigen::MatrixXd _eigenVectorsNormal;     //!< eigenvectors projected to vertex normal direction, dim: N_S * M
    Eigen::MatrixXd _normals;
    
    Eigen::VectorXd _qDot_c_plus;            //!< modal velocity
    Eigen::VectorXd _qDDot_c;                //!< modal acceleration
    
    ImpulseSeries _impulseSeries;

    Eigen::Matrix3d _Inertia;
    Eigen::Vector3d _centerOfMass;
    double _mass = 0.;

    Eigen::Matrix3d _I_Inv;

private:
    void CullNonSurfaceModes();
    double EffectiveMass(const Eigen::Vector3d& x, const Eigen::Vector3d& n);
    double EstimateContactTimeScale(int vertex_a, double contactSpeed, const Eigen::Vector3d& impulse_a);
    void ComputeInertia();
    void LoadImpulses();

    double _dt = 0.;
    std::string _dataPrefix;
    std::string _matName = "ABS Plastic";

    Eigen::VectorXd _eigenValues;        // eigenvalues from modal analysis (omega^2 * density)
    Eigen::MatrixXd _eigenVectors;       // dim: 3 * N_V * M before culling, 3 * N_S * M after culling

    Eigen::Matrix<double, Eigen::Dynamic, 3> _V;
    //Eigen::Matrix<double, Eigen::Dynamic, 3> _FN;
    Eigen::Matrix<int, Eigen::Dynamic, 3> _F;
    
    Eigen::VectorXd _curvature;

    Eigen::VectorXd _q_p;   //!< previous displacement
    Eigen::VectorXd _q_c;   //!< current displacement
    Eigen::VectorXd _q_n;   //!< next displacement
    Eigen::VectorXd _q_nn;  //!< next next displacement
};

} // namespace ModalSound

#endif // #ifndef MODAL_SOUND_H