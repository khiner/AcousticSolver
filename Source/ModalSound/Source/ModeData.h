/** (c) 2024 Kangrui Xue
 *
 * \file ModeData.h
 * \brief Defines ModeData struct
 * 
 * Based on code by Jui-Hsien Wang (https://github.com/jhwang7628/openpbso)
 *
 * TODO: in general, this library is still pretty barebones (and in need of cleanup)
 */

#ifndef _MODE_DATA_H
#define _MODE_DATA_H

#define _USE_MATH_DEFINES   // needed for M_PI (in Visual Studio)
#include <math.h>

#include <fstream>
#include <iostream>

#include <vector>
#include <Eigen/Dense>


namespace ModalSound {

/**
 * \struct ModeData
 * \brief Stores/reads/writes modal displacement shapes and frequencies
 */
struct ModeData 
{
    std::vector<double> _omegaSquared;  //!< \brief eigenvalues produced by modal analysis
    std::vector<std::vector<double>> _modes;    //!< \brief modal displacement shapes

    int _N_modesAudible = -1;
    double _freqThresCache = 22100.;
    double _densityCache = -1;

    const std::vector<double> &mode(int modeIndex) { return _modes.at(modeIndex); }
    double omegaSquared(int modeIndex) const { return _omegaSquared.at(modeIndex); }
    int numModes() const { return _omegaSquared.size(); }
    int numDOF() const { return (numModes() > 0) ? _modes.at(0).size() : 0; }

    void read(const char* filename)
    {
        std::ifstream fin(filename, std::ios::binary);

        // Read the size of the problem and the number of modes
        int nDOF, nModes;
        fin.read((char*)&nDOF, sizeof(int));
        fin.read((char*)&nModes, sizeof(int));

        // Read the eigenvalues
        _omegaSquared.resize(nModes);
        fin.read((char *) _omegaSquared.data(), sizeof(double) * nModes);

        // Read the eigenvectors
        _modes.resize(nModes);
        for (int i = 0; i < nModes; i++) {
            _modes[i].resize(nDOF);
            fin.read((char*)_modes[i].data(), sizeof(double) * nDOF);
        }

        fin.close();
    }

    void write(const char* filename) const
    {
        std::ofstream fout(filename, std::ios::binary);

        int nModes = _omegaSquared.size();
        int nDOF;
        nDOF = _modes[0].size();
        fout.write((const char*)&nDOF, sizeof(int));
        fout.write((const char*)&nModes, sizeof(int));

        // Write the eigenvalues
        fout.write((const char*)_omegaSquared.data(), sizeof(double) * nModes);

        // Write the eigenvectors
        for (int i = 0; i < nModes; i++) {
            fout.write((const char*)_modes[i].data(), sizeof(double) * nDOF);
        }

        fout.close();
    }

    void printAllFrequency(const double& density) const
    {
        typedef typename std::vector<double>::const_iterator Iterator;
        int count = 0;
        for (Iterator it = _omegaSquared.begin(); it != _omegaSquared.end(); ++it, count++)
            printf("Mode %u: %f Hz\n", count, sqrt((*it) / density) / (2. * M_PI));
    }

    int numModesAudible(const double& density, const double& audibleFreq)
    {
        // use cache
        if (density == _densityCache && _freqThresCache == audibleFreq && _N_modesAudible >= 0) 
        {
            return _N_modesAudible;
        }
        auto Freq = [&](const double os)->double 
        {
            return sqrt(os / density) / (2. * M_PI);
        };
        if (_omegaSquared.size() == 0 || Freq(_omegaSquared.at(0)) > audibleFreq) 
        {
            return 0;
        }
        if (Freq(_omegaSquared.at(_omegaSquared.size() - 1)) <= audibleFreq) 
        {
            return _omegaSquared.size();
        }

        int ii; 
        for (ii = 0; ii < _omegaSquared.size(); ++ii) 
        {
            if (Freq(_omegaSquared.at(ii)) > audibleFreq) { break; }
        }
        _N_modesAudible = ii;
        _densityCache = density;
        _freqThresCache = audibleFreq;
        
        return _N_modesAudible;
    }
};

} // namespace ModalSound

#endif // _MODE_DATA_H