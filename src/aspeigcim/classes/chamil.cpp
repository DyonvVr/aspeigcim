#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "boost/math/special_functions/binomial.hpp"

class CHamil
{
private:
    int
        spin_modes, // number of spin states per "spatial" mode
        k, // number of spatial orbitals
        K, // number of spin-spatial orbitals
        N, // number of electrons
        d; // Fock space dimension
    std::vector<std::vector<int>>
        dets;
    std::vector<std::vector<std::vector<std::array<int, 5>>>>
        at_at_a_a;

    double get4DTensorElem(double *T_p, int p, int q, int r, int s) const
    {
        return T_p[p * k * k * k + q * k * k + r * k + s];
    }

    std::vector<std::vector<int>> allCombinations(std::vector<int> elems)
    {
        std::vector<int> comb;
        std::vector<std::vector<int>> list;
        combRecurse(comb, list, elems, 0, N);
        return list;
    }

    void combRecurse(std::vector<int> &comb, std::vector<std::vector<int>> &list, std::vector<int> elems,
                     int offset, int n)
    {
        if(n == 0)
            list.push_back(comb);
        else
            for(int i = offset; i < K - n + 1; ++i)
            {
                comb.push_back(elems[i]);
                combRecurse(comb, list, elems, i + 1, n - 1);
                comb.pop_back();
            }
    }

    // finds all spatial orbitals present in determinant
    std::vector<int> spatialOrbitalsInDet(const std::vector<int>& det) const
    {
        std::vector<int> spatialOrbitals;
        int p_prev = -1;

        for(auto const &p_sigma: det)
        {
            int p = (int)(p_sigma / spin_modes);
            if(p != p_prev)
            {
                int x = 0;
                spatialOrbitals.push_back(p); // uses that each p occurs at most twice, one for each spin
            }
            p_prev = p;
        }

        return spatialOrbitals;
    }

    int findCreationAnnihilationQuartetElement(int I, int J, int p, int r, int s, int q)
    {
        std::vector<std::array<int, 5>> at_at_a_a_IJ = at_at_a_a[I][J];

        for(std::vector<int>::size_type i = 0; i != at_at_a_a_IJ.size(); i++)
        {
            std::array<int, 5> arr = at_at_a_a_IJ[i];

            if(arr[0] > p) return 0;  // p values sorted in ascending order, so we won't find this combination after here
            if(arr[0] == p)
            {
                if(arr[1] > r) return 0;
                if(arr[1] == r)
                {
                    if(arr[2] > s) return 0;
                    if(arr[2] == s)
                    {
                        if(arr[3] > q) return 0;
                        if(arr[3] == q) return arr[4];  // arr[4] stores sign
                    }
                }
            }
        }

        return 0;
    }

    void buildCreationAnnihilationQuartetElements()
    {
        if(spin_modes == 1)
            buildCreationAnnihilationQuartetElementsSpinless();
        else
            buildCreationAnnihilationQuartetElementsSpin();
    }

    // compute elements <I|a†_p * a†_r * a_s * a_q|J>
    // where I, J are determinant indices
    void buildCreationAnnihilationQuartetElementsSpinless()
    {
        int progressPercentage;
        int progressPercentagePrev = -1;

        for(int I = 0; I < d; I++)
        {
            std::vector<std::vector<std::array<int, 5>>> at_at_a_a_I;

            for(int J = 0; J < d; J++)
            {
                progressPercentage = (int)(100 * (I * d + J) / (d * d));
                if(progressPercentage > progressPercentagePrev)
                    std::cout << "\r" << "Building excitation operators... (" << progressPercentage << "%)   ";
                progressPercentagePrev = progressPercentage;

                std::vector<std::array<int, 5>> at_at_a_a_IJ;

                for(auto const &p: dets[I]) for(auto const &r: dets[I]) for(auto const &q: dets[J]) for(auto const &s: dets[J])
                {
                    //if(p >= r || q >= s) continue;

                    int elem = creationAnnihilationQuartetElement(p, r, s, q, I, J);
                    if(elem != 0)
                    {
                        std::array<int, 5> apt_art_as_aq_IJ = {p, r, s, q, elem};
                        at_at_a_a_IJ.push_back(apt_art_as_aq_IJ);
                    }
                }

                at_at_a_a_I.push_back(at_at_a_a_IJ);
            }

            at_at_a_a.push_back(at_at_a_a_I);
        }

        std::cout << "\rBuilding excitation operators... (100%)    " << std::endl << "Done" << std::endl << std::endl << std::flush;
    }

    // builds the two-electron excitation operators
    // e_{prsq} = \sum_{\sigma\tau}a\t_{p\sigma}a\t_{r\tau}a_{s\tau}a_{q\sigma}
    // see Helgaker eq. 2.2.16 (but note different index ordering in their book)
    void buildCreationAnnihilationQuartetElementsSpin()
    {
        int progressPercentage;
        int progressPercentagePrev = -1;

        for(int I = 0; I < d; I++)
        {
            std::vector<std::vector<std::array<int, 5>>> at_at_a_a_I;

            for(int J = 0; J < d; J++)
            {
                progressPercentage = (int)(100 * (I * d + J) / (d * d));
                if(progressPercentage > progressPercentagePrev)
                    std::cout << "\r" << "Building excitation operators... (" << progressPercentage << "%)   ";
                progressPercentagePrev = progressPercentage;

                std::vector<std::array<int, 5>> at_at_a_a_IJ;

                std::vector<int> spatialOrbitalsI = spatialOrbitalsInDet(dets[I]);
                std::vector<int> spatialOrbitalsJ = spatialOrbitalsInDet(dets[J]);

                for(auto const &p : spatialOrbitalsI) for(auto const &r : spatialOrbitalsI)
                for(auto const &s : spatialOrbitalsJ) for(auto const &q : spatialOrbitalsJ)
                {
                    int orb_elem = 0;

                    for(int sigma = 0; sigma < spin_modes; sigma++)  for(int tau = 0; tau < spin_modes; tau++)
                    {
                        int p_sigma = spin_modes * p + sigma;
                        int q_sigma = spin_modes * q + sigma;
                        int r_tau   = spin_modes * r + tau;
                        int s_tau   = spin_modes * s + tau;

                        int spin_orb_elem = creationAnnihilationQuartetElement(p_sigma, r_tau, s_tau, q_sigma, I, J);
                        orb_elem += spin_orb_elem;
                    }

                    if(orb_elem != 0)
                    {
                        std::array<int, 5> apt_art_as_aq_IJ = {p, r, s, q, orb_elem};
                        at_at_a_a_IJ.push_back(apt_art_as_aq_IJ);
                    }
                }

                at_at_a_a_I.push_back(at_at_a_a_IJ);
            }

            at_at_a_a.push_back(at_at_a_a_I);
        }

        std::cout << "\rBuilding excitation operators... (100%)    " << std::endl << "Done" << std::endl << std::endl << std::flush;
    }

    // compute element <I|a†_p\sigma * a†_r\tau * a_s\tau * a_q\sigma|J>
    // where I, J are determinant indices
    int creationAnnihilationQuartetElement(int p_sigma, int r_tau, int s_nu, int q_mu, int I, int J)
    {
        auto p_sigma_r_tauInDetI = extractOrbitalPairFromDet(p_sigma, r_tau, dets[I]);
        auto q_mu_s_nuInDetJ = extractOrbitalPairFromDet(q_mu, s_nu, dets[J]);

        if(p_sigma_r_tauInDetI.first == 0 || q_mu_s_nuInDetJ.first == 0) return 0;

        int p_sigma_r_tauSign = p_sigma_r_tauInDetI.first;
        int q_mu_s_nuSign = q_mu_s_nuInDetJ.first;
        auto detIStripped = p_sigma_r_tauInDetI.second;
        auto detJStripped = q_mu_s_nuInDetJ.second;

        return p_sigma_r_tauSign * q_mu_s_nuSign * compareDets(detIStripped, detJStripped);
    }

    std::pair<int, std::vector<int>> extractOrbitalPairFromDet(int p_sigma, int r_tau, std::vector<int> det)
    {
        // routine assumes that p_sigma < r_tau, i.e. spin orbital p_sigma (if present) occurs to the left of
        // spin orbital r_tau (if present)

        std::pair<int, std::vector<int>> result;
        int sign = 1;
        std::vector<int> detStripped = det; // copy; to be determinant without p and r orbitals

        // find spin orbital p_sigma in detStripped
        auto it_p_sigma = std::find(detStripped.begin(), detStripped.end(), p_sigma);
        if (it_p_sigma != detStripped.end()) // spin orbital p_sigma present
        {
            int index_p_sigma = std::distance(detStripped.begin(), it_p_sigma); // index of p_sigma in detStripped
            sign *= pow(-1, index_p_sigma);
            detStripped.erase(detStripped.begin() + index_p_sigma);
        }
        else
        {
            result.first = 0;
            return result;
        }

        // find spin orbital r_tau in detStripped
        auto it_r_tau = std::find(detStripped.begin(), detStripped.end(), r_tau);
        if (it_r_tau != detStripped.end()) // spin orbital r_tau present
        {
            int index_r_tau = std::distance(detStripped.begin(), it_r_tau); // index of r in detStripped
            sign *= pow(-1, index_r_tau); // notice p_sigma < r_tau, so one orbital to the left of r has already been removed
            detStripped.erase(detStripped.begin() + index_r_tau);
        }
        else
        {
            result.first = 0;
            return result;
        }

        result.first = sign;
        result.second = detStripped;

        return result;
    }

    // compares two determinants; returns 1 only if all elements are equal, 0 otherwise
    // uses that determinants are sorted
    int compareDets(std::vector<int> det1, std::vector<int> det2)
    {
        if(det1.size() != det2.size()) return 0;

        for(int i = 0; i < det1.size(); i++)
            if(det1[i] != det2[i]) return 0;

        return 1;
    }

public:
    CHamil(int k, int N, int spin_modes) : k(k), N(N), spin_modes(spin_modes)
    {
        K = k * spin_modes;
        d = (int)boost::math::binomial_coefficient<double>(K, N);

        std::vector<int> orbitalIndices;
        for(int i = 0; i < K; i++) orbitalIndices.push_back(i);
        dets = allCombinations(orbitalIndices);

        buildCreationAnnihilationQuartetElements();
    }

    int getFockSpaceDim() const
    {
        return d;
    }

    void show4DTensor(double *T_p)
    {
        for(int p = 0; p < k; p++) for(int q = 0; q < k; q++) for(int r = 0; r < k; r++) for(int s = 0; s < k; s++)
        {
            double T_pqrs = get4DTensorElem(T_p, p, q, r, s);
            printf("T[%d, %d, %d, %d] = %f\n", p, q, r, s, T_pqrs);
        }
    }

    void populateHamiltonian(double *H_p, double *G_p)
    {
        if(spin_modes == 1)
            populateHamiltonianSpinless(H_p, G_p);
        else
            populateHamiltonianSpin(H_p, G_p);
    }

    void populateHamiltonianSpinless(double *H_p, double *G_p)
    {
        for(int I = 0; I < d; I++) for(int J = 0; J < d; J++)
        {
            for(auto const &p: dets[I]) for(auto const &r: dets[I]) for(auto const &q: dets[J]) for(auto const &s: dets[J])
            {
//                if(p >= r || q >= s) continue;

                double weight_G;
//                weight_G = 2 * (get4DTensorElem(G_p, p, q, r, s) - get4DTensorElem(G_p, r, q, p, s));
                weight_G = get4DTensorElem(G_p, p, q, r, s);

                if(weight_G <= 1.0e-14 && weight_G >= -1.0e-14) continue;

                int apt_art_as_aq_IJ = findCreationAnnihilationQuartetElement(I, J, p, r, s, q);

                double contribution = weight_G * apt_art_as_aq_IJ;

                H_p[d * I + J] += contribution;
            }
        }
    }

    void populateHamiltonianSpin(double *H_p, double *G_p)
    {
        for(int I = 0; I < d; I++) for(int J = 0; J < d; J++)
        {
            auto spatialOrbitalsI = spatialOrbitalsInDet(dets[I]);
            auto spatialOrbitalsJ = spatialOrbitalsInDet(dets[J]);

            for(auto const &p: spatialOrbitalsI) for(auto const &r: spatialOrbitalsI)
            for(auto const &q: spatialOrbitalsJ) for(auto const &s: spatialOrbitalsJ)
            {
                double G_pqrs = get4DTensorElem(G_p, p, q, r, s);
                if(G_pqrs <= 1.0e-14 && G_pqrs >= -1.0e-14) continue;

                int e_prsq_IJ = findCreationAnnihilationQuartetElement(I, J, p, r, s, q);

                H_p[d * I + J] += G_pqrs * e_prsq_IJ;
            }
        }
    }

    int run()
    {
        return 0;
    }
};

extern "C"
{
CHamil *CHamil_new(int K, int N, int spin_modes)
{
    return new CHamil(K, N, spin_modes);
}
void CHamil_show4DTensor(CHamil *chamil, double *T_p)
{
    chamil->show4DTensor(T_p);
}
void CHamil_populateHamiltonian(CHamil *chamil, double *H_p, double *G_p)
{
    chamil->populateHamiltonian(H_p, G_p);
}
}

int main()
{
    int k = 6;
    int spin_modes = 1;
    int N = 3;

    auto *G_p = (double *)malloc(sizeof(double) * k * k * k * k);
    for(int p = 0; p < k; p++) for(int r = 0; r < k; r++) for(int q = 0; q < k; q++) for(int s = 0; s < k; s++)
    {
        int address = p * k * k * k + q * k * k + r * k + s;
        G_p[address] = 0;

        if(r == s)
        {
            if((p == 0 && q == 1) || (p == 1 && q == 0) || (p == 2 && q == 3) || (p == 3 && q == 2) || (p == 4 && q == 5) || (p == 5 && q == 4))
            {
                G_p[p * k * k * k + q * k * k + r * k + s] = 1.0 / (N - 1);
            }
        }
    }

    CHamil chamil(k, N, spin_modes);

    int d = chamil.getFockSpaceDim();
    auto *H_p = (double *)malloc(d * d * sizeof(double));
    chamil.populateHamiltonian(H_p, G_p);

    std::cout << "H = " << std::endl;
    for(int i = 0; i < d; i++)
    {
        for(int j = 0; j < d; j++)
        {
            std::cout << H_p[d * i + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}