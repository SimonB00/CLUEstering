
#pragma once

//
//	Author: Simone Balducci
//

#include "OneToManyAssociator.h"
#include "span.h"

class Followers {
private:
  OneToManyAssociator<int32_t, 1 << 27, 1 << 27> m_followers;

public:
  Followers() = default;

  ALPAKA_FN_HOST_ACC inline constexpr const uint32_t* offset() const {
    return m_followers.off.data();
  }
  ALPAKA_FN_HOST_ACC inline constexpr uint32_t* offset() {
    return m_followers.off.data();
  }
  ALPAKA_FN_HOST_ACC inline constexpr const uint32_t* offset(int32_t i) const {
    return m_followers.off.data() + i;
  }
  ALPAKA_FN_HOST_ACC inline constexpr uint32_t* offset(int32_t i) {
    return m_followers.off.data() + i;
  }

  ALPAKA_FN_HOST_ACC inline constexpr const int32_t* content() const {
    return m_followers.content.data();
  }
  ALPAKA_FN_HOST_ACC inline constexpr int32_t* content() {
    return m_followers.content.data();
  }
  ALPAKA_FN_HOST_ACC inline constexpr const int32_t* content(int32_t i) const {
    return m_followers.content.data() + i;
  }
  ALPAKA_FN_HOST_ACC inline constexpr int32_t* content(int32_t i) {
    return m_followers.content.data() + i;
  }

  ALPAKA_FN_HOST_ACC inline constexpr auto size(int32_t i) { return m_followers.size(i); }

  ALPAKA_FN_HOST_ACC inline constexpr clue::span<const int32_t> operator[](
      int32_t i) const {
    return clue::span<const int32_t>{m_followers.begin(i),
                                     m_followers.off[i + 1] - m_followers.off[i]};
  }
};
