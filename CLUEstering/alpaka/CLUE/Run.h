#ifndef run_h
#define run_h

#include <vector>

#include "CLUEAlgoAlpaka.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <uint8_t Ndim, typename Kernel>
  std::vector<std::vector<int>> run(const std::vector<float>& dc,
									const std::vector<float>& dm,
                                    float rhoc,
                                    int pPBin,
                                    const std::vector<std::vector<float>>& coordinates,
                                    const std::vector<float>& weight,
                                    const Kernel& kernel,
                                    Queue queue_,
                                    size_t block_size) {
	std::cout << __LINE__ << std::endl;
    CLUEAlgoAlpaka<Acc1D, Ndim> algo(dc, dm, rhoc, pPBin, queue_);
	std::cout << __LINE__ << std::endl;

    // Create the host and device points
    Points<Ndim> h_points(coordinates, weight);
    PointsAlpaka<Ndim> d_points(queue_, weight.size());

	std::cout << __LINE__ << std::endl;
    return algo.make_clusters(h_points, d_points, kernel, queue_, block_size);
  }

};  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
