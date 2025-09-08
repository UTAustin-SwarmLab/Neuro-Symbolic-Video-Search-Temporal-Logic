/**
 * @file
 * A collection of functions to get the roots of univariate polynomials
 * whose coeffecients may be multivariate polynomials, after replacing
 * the variables in the coefficients by specific values.
 *
 * @ingroup rootfinder
 */


#pragma once

#include "RootFinder.h"

#include "../../formula/model/ran/RealAlgebraicNumberEvaluation.h"

namespace carl {
namespace rootfinder {


// hiervon eine thom version machen!!!
template<typename Coeff, typename Number>
boost::optional<std::vector<RealAlgebraicNumber<Number>>> realRoots(
		const UnivariatePolynomial<Coeff>& poly,
		const std::map<Variable, RealAlgebraicNumber<Number>>& varToRANMap,
		const Interval<Number>& interval,
		SplittingStrategy pivoting
) {
	CARL_LOG_FUNC("carl.core.rootfinder", poly << " in " << poly.mainVar() << ", " << varToRANMap << ", " << interval);
	assert(varToRANMap.count(poly.mainVar()) == 0);

	if (poly.isZero()) {
		CARL_LOG_TRACE("carl.core.rootfinder", "poly is 0 -> everything is a root");
		return boost::none;
	}
	if (poly.isNumber()) {
		CARL_LOG_TRACE("carl.core.rootfinder", "poly is constant but not zero -> no root");
		return std::vector<RealAlgebraicNumber<Number>>({});
	}

  // We want to simplify 'poly', but it's const, so make a copy.
	UnivariatePolynomial<Coeff> polyCopy(poly);
	std::map<Variable, RealAlgebraicNumber<Number>> IRmap;

	for (Variable v: polyCopy.gatherVariables()) {
		if (v == poly.mainVar()) continue;
		if (varToRANMap.count(v) == 0) {
			CARL_LOG_TRACE("carl.core.rootfinder", "poly still contains unassigned variable " << v);
			return boost::none;
		}
		assert(varToRANMap.count(v) > 0);
		if (varToRANMap.at(v).isNumeric()) {
			polyCopy.substituteIn(v, Coeff(varToRANMap.at(v).value()));
		} else {
			IRmap.emplace(v, varToRANMap.at(v));
		}
	}
	if (polyCopy.isZero()) {
		CARL_LOG_TRACE("carl.core.rootfinder", "poly is 0 after substituting rational assignments -> everything is a root");
		return boost::none;
	}
	if (IRmap.empty()) {
		assert(polyCopy.isUnivariate());
		return realRoots(polyCopy, interval, pivoting);
	} else {
		CARL_LOG_TRACE("carl.core.rootfinder", poly << " in " << poly.mainVar() << ", " << varToRANMap << ", " << interval);
		assert(IRmap.find(polyCopy.mainVar()) == IRmap.end());
		UnivariatePolynomial<Number> evaledpoly = RealAlgebraicNumberEvaluation::evaluatePolynomial(polyCopy, IRmap);
		if (evaledpoly.isZero()) return boost::none;
		CARL_LOG_TRACE("carl.core.rootfinder", "Calling on " << evaledpoly);
		auto res = realRoots(evaledpoly, interval, pivoting);
		MultivariatePolynomial<Number> mvpoly(polyCopy);
		CARL_LOG_TRACE("carl.core.rootfinder", "Checking " << res << " on " << mvpoly);
		for (auto it = res.begin(); it != res.end();) {
			CARL_LOG_TRACE("carl.core.rootfinder", "Checking " << polyCopy.mainVar() << " = " << *it);
			IRmap[polyCopy.mainVar()] = *it;
			CARL_LOG_TRACE("carl.core.rootfinder", "Evaluating " << mvpoly << " on " << IRmap);
			if (!RealAlgebraicNumberEvaluation::evaluate(mvpoly, IRmap).isZero()) {
				CARL_LOG_TRACE("carl.core.rootfinder", "Purging spurious root " << *it);
				it = res.erase(it);
			} else {
				it++;
			}
		}
		return res;
	}
}

template<typename Coeff, typename Number>
boost::optional<std::vector<RealAlgebraicNumber<Number>>> realRoots(
		const UnivariatePolynomial<Coeff>& poly,
		const std::list<Variable>& variables,
		const std::list<RealAlgebraicNumber<Number>>& values,
		const Interval<Number>& interval,
		SplittingStrategy pivoting
) {
	std::map<Variable, RealAlgebraicNumber<Number>> varToRANMap;

	assert(variables.size() == values.size());
	auto varit = variables.begin();
	auto valit = values.begin();
	while (varit != variables.end()) {
		varToRANMap[*varit] = *valit;
		varit++;
		valit++;
	}
	return realRoots(poly, varToRANMap, interval, pivoting);
}

}
}
