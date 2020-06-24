#pragma once

#include <algorithm>
#include <iterator>

/////////////////////////////////////////////////////
//////                                         //////
//////    HELPER TEMPLATE ALGORITHMS           //////
//////                                         //////
/////////////////////////////////////////////////////

template<typename OuterIterTy,
         typename InnerIterTy,
         typename BinaryPredicate,
         typename OuterValueTy =
           typename std::iterator_traits<OuterIterTy>::value_type,
         typename InnerValueTy =
           typename std::iterator_traits<InnerIterTy>::value_type>
static constexpr auto
any_of_range(const OuterIterTy outer_iter_begin,
             const OuterIterTy outer_iter_end,
             const InnerIterTy inner_iter_begin,
             const InnerIterTy inner_iter_end,
             const BinaryPredicate& pred) -> bool
{
  return std::all_of(
    outer_iter_begin, outer_iter_end, [&](const OuterValueTy& outer_value) {
      return std::any_of(
        inner_iter_begin, inner_iter_end, [&](const InnerValueTy& inner_value) {
          return pred(outer_value, inner_value);
        });
    });
}

template<typename OuterIterTy,
         typename InnerIterTy,
         typename OuterValueTy =
           typename std::iterator_traits<OuterIterTy>::value_type,
         typename InnerValueTy =
           typename std::iterator_traits<InnerIterTy>::value_type>
static auto
any_of_range(const OuterIterTy outer_iter_begin,
             const OuterIterTy outer_iter_end,
             const InnerIterTy inner_iter_begin,
             const InnerIterTy inner_iter_end) -> bool
{
  return any_of_range(outer_iter_begin,
                      outer_iter_end,
                      inner_iter_begin,
                      inner_iter_end,
                      std::equal_to{});
}
