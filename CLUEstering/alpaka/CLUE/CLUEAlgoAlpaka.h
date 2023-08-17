#ifndef CLUE_Algo_Alpaka_h
#define CLUE_Algo_Alpaka_h

#include <algorithm>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/vec/Vec.hpp>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdint.h>
#include <string>
#include <vector>
#include <utility>

#include "CLUEAlpakaKernels.h"
#include "../DataFormats/Points.h"
#include "../DataFormats/alpaka/PointsAlpaka.h"
#include "../DataFormats/alpaka/TilesAlpaka.h"
#include "../DataFormats/Math/DeltaPhi.h"
#include "ConvolutionalKernel.h"

using cms::alpakatools::VecArray;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  /* template <uint8_t Ndim> */
  /* class domain_t { */
  /* private: */
	/* VecArray<VecArray<float, 2>, Ndim> domains_; */

  /* public: */
	/* domain_t(const std::array<std::pair<float, float>, Ndim>& domains) { */
	  /* for (int dim{}; dim < Ndim; ++dim) { */
		/* VecArray<float, 2> temp; */
		/* temp.push_back_unsafe(domains[dim][0]); */
		/* temp.push_back_unsafe(domains[dim][1]); */

		/* domains_.push_back_unsafe(temp); */
	  /* } */
	/* } */

	/* // Getters */
	/* VecArray<VecArray<float, 2>, Ndim> data() { */
	  /* return domains_; */
	/* } */
	/* VecArray<VecArray<float, 2>, Ndim>* get() { */
	  /* return domains_[0].begin(); */
	/* } */
  /* }; */

  template <typename TAcc, uint8_t Ndim>
  class CLUEAlgoAlpaka {
  public:
    CLUEAlgoAlpaka() = delete;
    explicit CLUEAlgoAlpaka(float dc, float rhoc, float outlierDeltaFactor, int pPBin, Queue queue_)
        : dc_{dc}, rhoc_{rhoc}, outlierDeltaFactor_{outlierDeltaFactor}, pointsPerTile_{pPBin} {
      init_device(queue_);
    }

    TilesAlpaka<Ndim>* m_tiles;
    VecArray<int32_t, max_seeds>* m_seeds;
    VecArray<int32_t, max_followers>* m_followers;

    std::vector<std::vector<int>> make_clusters(Points<Ndim>& h_points,
                                                PointsAlpaka<Ndim>& d_points,
                                                const ConvolutionalKernel& kernel,
                                                Queue queue_);

  private:
    float dc_;
    float rhoc_;
    float outlierDeltaFactor_;
    // average number of points found in a tile
    int pointsPerTile_;

	/* domain_t<Ndim> m_domains; */

    // Buffers
    std::optional<cms::alpakatools::device_buffer<Device, TilesAlpaka<Ndim>>> d_tiles;
    std::optional<cms::alpakatools::device_buffer<Device, cms::alpakatools::VecArray<int32_t, max_seeds>>>
        d_seeds;
    std::optional<
        cms::alpakatools::device_buffer<Device, cms::alpakatools::VecArray<int32_t, max_followers>[]>>
        d_followers;

    // Private methods
    void init_device(Queue queue_);
    void setup(const Points<Ndim>& h_points, PointsAlpaka<Ndim>& d_points, Queue queue_);

    // Construction of the tiles
    int calculateNTiles(const Points<Ndim>& h_points);
    VecArray<float, Ndim> calculate_tile_size(int nTiles, const Points<Ndim>& h_points);
  };

  // Private methods
  template <typename TAcc, uint8_t Ndim>
  int CLUEAlgoAlpaka<TAcc, Ndim>::calculateNTiles(const Points<Ndim>& h_points) {
    size_t n_tiles{h_points.n / pointsPerTile_};
    try {
      if (n_tiles == 0) {
        throw 100;
      }
    } catch (...) {
      std::cout << "pointPerBin is set too high for you number of points. You must lower it in the "
                   "clusterer constructor.\n";
    }
    return n_tiles;
  }

  template <typename TAcc, uint8_t Ndim>
  VecArray<float, Ndim> CLUEAlgoAlpaka<TAcc, Ndim>::calculate_tile_size(int nTiles,
                                                                        const Points<Ndim>& h_points) {
    VecArray<float, Ndim> tileSizes;
    int NperDim{static_cast<int>(std::pow(nTiles, 1.0 / Ndim))};

    for (int i{}; i != Ndim; ++i) {
      float tileSize;
      float dimMax{*std::max_element(h_points.m_coords[i].begin(), h_points.m_coords[i].end())};
      float dimMin{*std::min_element(h_points.m_coords[i].begin(), h_points.m_coords[i].end())};

      VecArray<float, 2> temp;
      temp.push_back_unsafe(dimMin);
      temp.push_back_unsafe(dimMax);
      m_tiles->min_max[i] = temp;
      tileSize = (dimMax - dimMin) / NperDim;

      tileSizes[i] = tileSize;
    }
    return tileSizes;
  }

  template <typename TAcc, uint8_t Ndim>
  void CLUEAlgoAlpaka<TAcc, Ndim>::init_device(Queue queue_) {
    d_tiles = cms::alpakatools::make_device_buffer<TilesAlpaka<Ndim>>(queue_);
    d_seeds = cms::alpakatools::make_device_buffer<cms::alpakatools::VecArray<int32_t, max_seeds>>(queue_);
    d_followers =
        cms::alpakatools::make_device_buffer<cms::alpakatools::VecArray<int32_t, max_followers>[]>(queue_,
                                                                                                   reserve);

    // Copy to the public pointers
    m_tiles = (*d_tiles).data();
    m_seeds = (*d_seeds).data();
    m_followers = (*d_followers).data();
  }

  template <typename TAcc, uint8_t Ndim>
  void CLUEAlgoAlpaka<TAcc, Ndim>::setup(const Points<Ndim>& h_points,
                                         PointsAlpaka<Ndim>& d_points,
                                         Queue queue_) {
    m_tiles->n_tiles = calculateNTiles(h_points);
    m_tiles->resizeTiles();
    m_tiles->tile_size = calculate_tile_size(m_tiles->n_tiles, h_points);

    alpaka::memcpy(
        queue_, d_points.coords, cms::alpakatools::make_host_view(h_points.m_coords.data(), h_points.n));
    alpaka::memcpy(
        queue_, d_points.weight, cms::alpakatools::make_host_view(h_points.m_weight.data(), h_points.n));
    alpaka::memset(queue_, (*d_seeds), 0x00);

    // Define the working division
    const Idx block_size{1024};
    Idx grid_size = std::ceil(h_points.n / static_cast<float>(block_size));
    auto working_div = cms::alpakatools::make_workdiv<Acc1D>(grid_size, block_size);
    alpaka::enqueue(
        queue_,
        alpaka::createTaskKernel<Acc1D>(working_div, KernelResetFollowers{}, m_followers, h_points.n));
  }

  // Public methods
  template <typename TAcc, uint8_t Ndim>
  std::vector<std::vector<int>> CLUEAlgoAlpaka<TAcc, Ndim>::make_clusters(Points<Ndim>& h_points,
                                                                          PointsAlpaka<Ndim>& d_points,
                                                                          const ConvolutionalKernel& kernel,
                                                                          Queue queue_) {
    setup(h_points, d_points, queue_);

    const Idx block_size{1024};
    const Idx grid_size = std::ceil(h_points.n / static_cast<float>(block_size));
    auto working_div = cms::alpakatools::make_workdiv<Acc1D>(grid_size, block_size);
    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(
                        working_div, KernelFillTiles(), d_points.view(), m_tiles, h_points.n));

    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(working_div,
                                                    KernelCalculateLocalDensity(),
                                                    m_tiles,
                                                    d_points.view(),
                                                    kernel,
													/* m_domains.data(), */
                                                    dc_,
                                                    h_points.n));
    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(working_div,
                                                    KernelCalculateNearestHigher(),
                                                    m_tiles,
                                                    d_points.view(),
													/* m_domains.data(), */
                                                    outlierDeltaFactor_,
                                                    dc_,
                                                    h_points.n));
    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(working_div,
                                                    KernelFindClusters<Ndim>(),
                                                    m_seeds,
                                                    m_followers,
                                                    d_points.view(),
                                                    outlierDeltaFactor_,
                                                    dc_,
                                                    rhoc_,
                                                    h_points.n));
    alpaka::enqueue(queue_,
                    alpaka::createTaskKernel<Acc1D>(
                        working_div, KernelAssignClusters<Ndim>(), m_seeds, m_followers, d_points.view()));

    // Wait for all the operations in the queue to finish
    alpaka::wait(queue_);

    auto rho_view = cms::alpakatools::make_host_view(h_points.m_rho.data(), h_points.n);
    alpaka::memcpy(queue_, rho_view, d_points.rho, static_cast<uint32_t>(h_points.n));
    alpaka::memcpy(queue_,
                   cms::alpakatools::make_host_view(h_points.m_delta.data(), h_points.n),
                   d_points.delta,
                   static_cast<uint32_t>(h_points.n));
    alpaka::memcpy(queue_,
                   cms::alpakatools::make_host_view(h_points.m_nearestHigher.data(), h_points.n),
                   d_points.nearest_higher,
                   static_cast<uint32_t>(h_points.n));
    alpaka::memcpy(queue_,
                   cms::alpakatools::make_host_view(h_points.m_clusterIndex.data(), h_points.n),
                   d_points.cluster_index,
                   static_cast<uint32_t>(h_points.n));
    alpaka::memcpy(queue_,
                   cms::alpakatools::make_host_view(h_points.m_isSeed.data(), h_points.n),
                   d_points.is_seed,
                   static_cast<uint32_t>(h_points.n));

    // Wait for all the operations in the queue to finish
    alpaka::wait(queue_);

    return {h_points.m_clusterIndex, h_points.m_isSeed};
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
#endif
