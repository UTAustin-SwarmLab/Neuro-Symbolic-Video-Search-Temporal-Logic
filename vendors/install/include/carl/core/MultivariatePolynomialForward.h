#pragma once
#include "MultivariatePolynomialPolicyForward.h"

namespace carl {
class Variable;

struct NotRelevant {};
template<typename Coeff, typename Ordering = NotRelevant, typename Policies = StdMultivariatePolynomialPolicies<>>
class MultivariatePolynomial;
}