#pragma once

/////////////////////////////////////////////////////
//////                                         //////
//////    HELPER TEMPLATE ALGORITHMS           //////
//////                                         //////
/////////////////////////////////////////////////////

template<typename OuterIterable,
         typename InnerIterable,
         typename BinaryPredicate>
static constexpr auto
any_of_range(const OuterIterable& outer_iterable,
             const InnerIterable& inner_iterable,
             const BinaryPredicate& pred) -> bool
{
  for (std::size_t x = 0; x < outer_iterable.size(); x++) {
    bool found = false;
    for (std::size_t y = 0; y < inner_iterable.size(); y++) {
      if (pred(outer_iterable[x], inner_iterable[y])) {
        found = true;
        break;
      }
    }
    if (!found) {
      return false;
    }
  }
  return true;
}

template<typename OuterIterable, typename InnerIterable>
static constexpr auto
any_of_range(const OuterIterable& outer_iterable,
             const InnerIterable& inner_iterable) -> bool
{
  return any_of_range(outer_iterable, inner_iterable, std::equal_to{});
}
