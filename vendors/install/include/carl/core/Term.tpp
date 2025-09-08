/**
 * @file Term.tpp
 * @ingroup MultiRP
 * @author Sebastian Junges
 */
#pragma once

#include "../numbers/numbers.h"
#include "Term.h"

namespace carl
{

template<typename Coefficient>
Term<Coefficient>::Term(const Coefficient& c) :
	mCoeff(c)
{
	assert(this->isConsistent());
}
template<typename Coefficient>
Term<Coefficient>::Term(Variable v) :
	mCoeff(carl::constant_one<Coefficient>().get()), mMonomial(createMonomial(v, uint(1)))
{
	assert(this->isConsistent());
}

template<typename Coefficient>
Term<Coefficient>::Term(Monomial::Arg m) :
	mCoeff(carl::constant_one<Coefficient>().get()), mMonomial(std::move(m))
{
	assert(this->isConsistent());
}

template<typename Coefficient>
Term<Coefficient>::Term(Monomial::Arg&& m) :
	mCoeff(carl::constant_one<Coefficient>().get()), mMonomial(std::move(m))
{
	assert(this->isConsistent());
}

template<typename Coefficient>
Term<Coefficient>::Term(const Coefficient& c, Monomial::Arg m) :
	mCoeff(c), mMonomial(std::move(m))
{
	assert(this->isConsistent());
}

template<typename Coefficient>
Term<Coefficient>::Term(Coefficient&& c, Monomial::Arg&& m) :
	mCoeff(std::move(c)), mMonomial(std::move(m))
{
	assert(this->isConsistent());
}

template<typename Coefficient>
Term<Coefficient>::Term(const Coefficient& c, Variable v, uint e): 
	mCoeff(c),
	mMonomial(createMonomial(v,e))
{
	assert(this->isConsistent());
}

template<typename Coefficient>
Term<Coefficient> Term<Coefficient>::divide(const Coefficient& c) const
{
	assert(!carl::isZero(c));
	return Term(mCoeff / c, mMonomial);
}

template<typename Coefficient>
bool Term<Coefficient>::divide(const Coefficient& c, Term& res) const
{
	assert(!carl::isZero(c));
	res.mCoeff = mCoeff / c;
	res.mMonomial = mMonomial;
	return true;
}

template<typename Coefficient>
bool Term<Coefficient>::divide(Variable v, Term& res) const
{
	if(mMonomial) {
		if(mMonomial->divide(v, res.mMonomial)) {
			res.mCoeff = mCoeff;
			return true;
		}
	}
	return false;
}

template<typename Coefficient>
bool Term<Coefficient>::divide(const Monomial::Arg& m, Term& res) const
{
	if (mMonomial) {
		if (mMonomial->divide(m, res.mMonomial)) {
			res.mCoeff = mCoeff;
			return true;
		}
		return false;
	}
	res = *this;
	return true;
}

template<typename Coefficient>
bool Term<Coefficient>::divide(const Term& t, Term& res) const
{
	assert(!carl::isZero(t.mCoeff));
	if (mMonomial) {
		if (t.mMonomial) {
			if (mMonomial->divide(t.mMonomial, res.mMonomial)) {
				res.mCoeff = mCoeff / t.mCoeff;
				return true;
			}
			return false;
		}
		res.mMonomial = mMonomial;
		res.mCoeff = mCoeff / t.mCoeff;
		return true;
	}
	if (t.mMonomial) return false;
	res.mMonomial = nullptr;
	res.mCoeff = mCoeff / t.mCoeff;
	return true;
}

template<typename Coefficient>
Term<Coefficient> Term<Coefficient>::derivative(Variable v) const
{
	if(!mMonomial)
	{
		// Derivatives of constants are zero.
		return Term<Coefficient>(carl::constant_zero<Coefficient>().get());
	}
	auto derivative = mMonomial->derivative(v);
	return Term<Coefficient>(Coefficient(mCoeff) * derivative.first, derivative.second);
}

template<typename Coefficient>
Definiteness Term<Coefficient>::definiteness() const
{
	if(mMonomial)
	{
		if(mMonomial->isSquare())
			return (mCoeff < CoefficientType(0) ? Definiteness::NEGATIVE_SEMI : Definiteness::POSITIVE_SEMI);
	}
	else if(!carl::isZero(mCoeff))    //change this to mCoeff.isZero() at some point
		return (mCoeff < CoefficientType(0) ? Definiteness::NEGATIVE : Definiteness::POSITIVE);
	return Definiteness::NON;
}

template<typename Coefficient>
Term<Coefficient> Term<Coefficient>::substitute(const std::map<Variable, Coefficient>& substitutions) const
{
	if (mMonomial) {
		Monomial::Content content;
		Coefficient coeff = mCoeff;
		for (const auto& c: *mMonomial) {
			auto it = substitutions.find(c.first);
			if (it == substitutions.end()) {
				content.push_back(c);
			} else {
				coeff *= carl::pow(it->second, c.second);
			}
		}
		if (content.empty()) return Term<Coefficient>(coeff);
		return Term<Coefficient>(coeff, createMonomial(std::move(content)));
	} else {
		return Term<Coefficient>(mCoeff);
	}
}

template<typename Coefficient>
Term<Coefficient> Term<Coefficient>::substitute(const std::map<Variable, Term<Coefficient>>& substitutions) const
{
	if (mMonomial) {
		return Coefficient(mCoeff) * mMonomial->substitute(substitutions);
	} else {
		return Term<Coefficient>(mCoeff);
	}
}

template<typename Coefficient>
Coefficient Term<Coefficient>::evaluate(const std::map<Variable, Coefficient>& map) const
{
	if (mMonomial) {
		return mCoeff * mMonomial->evaluate(map);
	} else {
		return mCoeff;
	}
}

template<typename Coefficient>
Term<Coefficient> Term<Coefficient>::calcLcmAndDivideBy(const Monomial::Arg& m) const
{
	assert(carl::isOne(coeff()));
	Monomial::Arg tmp = Monomial::calcLcmAndDivideBy(monomial(), m);
	if(tmp == nullptr)
	{
		return Term(carl::constant_one<Coefficient>().get());
	}
	else
	{
		return Term(carl::constant_one<Coefficient>().get(), tmp);
	}	
	

}

template<typename Coefficient>
bool Term<Coefficient>::sqrt(Term<Coefficient>& res) const {
    Coefficient resCoeff;
    if (!carl::sqrt_exact(this->coeff(), resCoeff)) return false;
    if (this->monomial() == nullptr) {
        res = Term(resCoeff);
        return true;
    }
    Monomial::Arg resMonomial = this->monomial()->sqrt();
    if (resMonomial == nullptr) return false;
    res = Term(resCoeff, resMonomial);
    return true;
}

template<typename Coefficient>
template<typename C, EnableIf<is_field<C>>>
bool Term<Coefficient>::divisible(const Term& t) const {
	if (this->monomial() == nullptr) return t.monomial() == nullptr;
	if (t.monomial() == nullptr) return true;
	return this->monomial()->divisible(t.monomial());
}

template<typename Coefficient>
template<typename C, DisableIf<is_field<C>>>
bool Term<Coefficient>::divisible(const Term& t) const {
	if (carl::remainder(this->coeff(), t.coeff()) != Coefficient(0)) return false;
	if (this->monomial() == nullptr) return t.monomial() == nullptr;
	if (t.monomial() == nullptr) return true;
	return this->monomial()->divisible(t.monomial());
}

template<typename Coefficient>
template<bool gatherCoeff, typename CoeffType>
void Term<Coefficient>::gatherVarInfo(Variable var, VariableInformation<gatherCoeff, CoeffType>& varinfo) const
{
	if(mMonomial)
	{
		varinfo.collect(var, coeff(), *mMonomial);
	}
	else
	{
		varinfo.updateCoeff( 0, *this );
	}
}

template<typename Coefficient>
template<bool gatherCoeff, typename CoeffType>
void Term<Coefficient>::gatherVarInfo(VariablesInformation<gatherCoeff, CoeffType>& varinfo) const
{
	if(mMonomial)
	{
		mMonomial->gatherVarInfo(varinfo, coeff());
	}
	else
	{
		// No change in varinfo.
	}
}


template<typename Coeff>
bool operator==(const Term<Coeff>& lhs, const Term<Coeff>& rhs) {
	if (lhs.coeff() != rhs.coeff()) return false;
	return lhs.monomial() == rhs.monomial();
}
template<typename Coeff>
bool operator==(const Term<Coeff>& lhs, const Monomial::Arg& rhs) {
	if (lhs.monomial() != rhs) return false;
	return carl::isOne(lhs.coeff());
}
template<typename Coeff>
bool operator==(const Term<Coeff>& lhs, Variable rhs) {
	if (lhs.monomial() != rhs) return false;
	return carl::isOne(lhs.coeff());
}
template<typename Coeff>
bool operator==(const Term<Coeff>& lhs, const Coeff& rhs) {
	return !lhs.monomial() && lhs.coeff() == rhs;
}

template<typename Coeff>
bool operator<(const Term<Coeff>& lhs, const Term<Coeff>& rhs) {
	if (lhs.monomial() == rhs.monomial()) return lhs.coeff() < rhs.coeff();
	return lhs.monomial() < rhs.monomial();
}

template<typename Coeff>
bool operator<(const Term<Coeff>& lhs, const Monomial::Arg& rhs) {
	if (lhs.monomial() == rhs) return lhs.coeff() < carl::constant_one<Coeff>().get();
	return lhs.monomial() < rhs;
}

template<typename Coeff>
bool operator<(const Term<Coeff>& lhs, Variable rhs) {
	if (lhs.monomial() == rhs) return lhs.coeff() < carl::constant_one<Coeff>().get();
	return lhs.monomial() < rhs;
}

template<typename Coeff>
bool operator<(const Term<Coeff>& lhs, const Coeff& rhs) {
	if (lhs.monomial() == nullptr) return lhs.coeff() < rhs;
	return false;
}

template<typename Coeff>
bool operator<(const Monomial::Arg& lhs, const Term<Coeff>& rhs) {
	if (lhs == rhs.monomial()) return carl::constant_one<Coeff>().get() < rhs.coeff();
	return lhs < rhs.monomial();
}

template<typename Coeff>
bool operator<(Variable lhs, const Term<Coeff>& rhs) {
	if (lhs == rhs.monomial()) return carl::constant_one<Coeff>().get() < rhs.coeff();
	return lhs < rhs.monomial();
}

template<typename Coeff>
bool operator<(const Coeff& lhs, const Term<Coeff>& rhs) {
	if (rhs.monomial() == nullptr) return lhs < rhs.coeff();
	return true;
}

template<typename Coeff>
const Term<Coeff> operator/(const Term<Coeff>& lhs, uint rhs)
{
	return Term<Coeff>(lhs.coeff()/carl::fromInt<Coeff>(rhs), lhs.monomial());
}

template<typename Coeff>
std::ostream& operator<<(std::ostream& os, const Term<Coeff>& rhs)
{
	return (os << rhs.toString(true, true));
}

template<typename Coeff>
std::ostream& operator<<(std::ostream& os, const std::shared_ptr<const Term<Coeff>>& rhs)
{
	if (rhs) return os << *rhs;
	else return os << "nullptr";
}

template<typename Coefficient>
template<typename C, DisableIf<is_interval<C>>>
std::string Term<Coefficient>::toString(bool infix, bool friendlyVarNames) const
{ 
	if(mMonomial)
	{
		if(mCoeff != CoefficientType(1))
		{
			std::stringstream s;
			if(!infix) s << " ";
			s << carl::toString(mCoeff,infix); //change to mCoeff.toString() at some point
			if(infix) return s.str() + "*" + mMonomial->toString(true, friendlyVarNames);
			else return "(*" + s.str() + " " + mMonomial->toString(infix, friendlyVarNames) + ")";
		}
		else
		{
			if(infix) return mMonomial->toString(true, friendlyVarNames);
			else return mMonomial->toString(infix, friendlyVarNames);
		}
	}
	else 
	{
		std::stringstream s;
		s << carl::toString(mCoeff, infix ); //change to mCoeff.toString()
		return s.str();
	}
}

template<typename Coefficient>
template<typename C, EnableIf<is_interval<C>>>
std::string Term<Coefficient>::toString(bool infix, bool friendlyVarNames) const
{ 
	if(mMonomial)
	{
		if(!mCoeff.isOne())
		{
			std::stringstream s;
			s << mCoeff;
			if(infix) return s.str() + "*" + mMonomial->toString(true, friendlyVarNames);
			else return "(*" + s.str() + " " + mMonomial->toString(infix, friendlyVarNames) + ")";
		}
		else
		{
			if(infix) return mMonomial->toString(true, friendlyVarNames);
			else return mMonomial->toString(infix, friendlyVarNames);
		}
	}
	else 
	{
		std::stringstream s;
		s << mCoeff;
		return s.str();
	}
}

template<typename Coefficient>
bool Term<Coefficient>::isConsistent() const {
	if (this->mMonomial) {
		assert(!this->mMonomial->isConstant());
	}
	return true;
}



template<typename Coefficient>
Term<Coefficient> Term<Coefficient>::gcd(const std::shared_ptr<const Term<Coefficient>>& t1, const std::shared_ptr<const Term<Coefficient>>& t2)
{
	static_assert(is_field<Coefficient>::value, "Not yet defined for other coefficients");
	assert(t1);
	assert(t2);
	return gcd(*t1, *t2);
	
}

template<typename Coefficient>
Term<Coefficient> Term<Coefficient>::gcd(const Term<Coefficient>& t1, const Term<Coefficient>& t2)
{
	static_assert(is_field<Coefficient>::value, "Not yet defined for other coefficients");
	assert(!t1.isZero());
	assert(!t2.isZero());
	if(t1.isConstant() || t2.isConstant()) return Term(Coefficient(carl::gcd(t1.coeff(), t2.coeff())));
	return Term(Coefficient(carl::gcd(t1.coeff(), t2.coeff())), Monomial::gcd(t1.monomial(), t2.monomial()));
}


}
