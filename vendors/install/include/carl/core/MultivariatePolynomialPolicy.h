/**
 * @file MultivariatePolynomialPolicy.h 
 * @ingroup multirp
 * @author Sebastian Junges
 */

#pragma once


#include "MultivariatePolynomialPolicyForward.h"
// TODO REMOVE FOLLOWING
#include "MonomialOrdering.h"



namespace carl
{
    /**
     * The default policy for polynomials. 
	 * @ingroup multirp
     */
    template<int>
	struct StdMultivariatePolynomialPolicies
    {
		
		
        /**
         * Linear searching means that we search linearly for a term instead of applying e.g. binary search.
         * Although the worst-case complexity is worse, for polynomials with a small nr of terms, this should be better.
         */
        static const bool searchLinear = true;
		static const bool has_reasons = false;

		virtual ~StdMultivariatePolynomialPolicies() = default;
    };
	
}
