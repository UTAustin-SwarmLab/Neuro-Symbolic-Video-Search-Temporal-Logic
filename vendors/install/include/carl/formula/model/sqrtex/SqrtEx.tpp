/**
 * Class to create a square root expression object.
 * @author Florian Corzilius
 * @since 2011-05-26
 * @version 2013-10-07
 */

namespace carl
{
    using namespace std;

	template<typename Poly>
    SqrtEx<Poly>::SqrtEx():
        mConstantPart(),
        mFactor(),
        mDenominator(1),
        mRadicand()
    {}

	template<typename Poly>
    SqrtEx<Poly>::SqrtEx( Poly&& _poly ):
        mConstantPart( std::move( _poly ) ),
        mFactor(),
        mDenominator(1),
        mRadicand()
    {
        normalize();
    }

	template<typename Poly>
    SqrtEx<Poly>::SqrtEx( Poly&& _constantPart, Poly&& _factor, Poly&& _denominator, Poly&& _radicand ):
        mConstantPart( std::move( _constantPart ) ),
        mFactor( _radicand.isZero() ? std::move( _radicand ) : std::move( _factor ) ),
        mDenominator( (mFactor.isZero() && mConstantPart.isZero()) ? constant_one<Poly>::get() : std::move( _denominator ) ),
        mRadicand( mFactor.isZero() ? mFactor : std::move( _radicand ) )
    {
        assert( !mDenominator.isZero() );
//        if( !( !mRadicand.isConstant() || mRadicand.isZero() || smtrat::ZERO_RATIONAL <= mRadicand.trailingTerm().coeff() ) )
//            exit(1569);
        assert( !mRadicand.isConstant() || mRadicand.isZero() || constant_zero<Rational>::get() <= mRadicand.trailingTerm().coeff() );
        normalize();
    }

	template<typename Poly>
    void SqrtEx<Poly>::normalize()
    {
//        std::cout << *this << std::endl;
        Poly gcdA;
        if( mFactor.isZero() )
        {
            gcdA = mConstantPart;
        }
        else 
        {
            Poly sqrtOfRadicand;
            if( mRadicand.sqrt( sqrtOfRadicand ) )
            {
                mConstantPart += mFactor * sqrtOfRadicand;
                mFactor = constant_zero<Poly>::get();
                mRadicand = constant_zero<Poly>::get();
            }
            else
            {
                assert( !radicand().isZero() );
                Rational absOfLCoeff = abs( radicand().coprimeFactor() );
                Rational sqrtResult = 0;
                if( carl::sqrt_exact( absOfLCoeff, sqrtResult ) )
                {
                    mFactor *= constant_one<Rational>::get()/sqrtResult;
                    mRadicand *= absOfLCoeff;
                }
            }
            if( mFactor.isZero() )
            {
                gcdA = mConstantPart;
            }
            else
            {
                if( mConstantPart.isZero() )
                {
                    gcdA = mFactor;
                }
                else
                {
                    Rational ccConstantPart = mConstantPart.coprimeFactor();
                    Poly cpConstantPart = mConstantPart * ccConstantPart;
                    Rational ccFactor = mFactor.coprimeFactor();
                    Poly cpFactor = mFactor * ccFactor;
                    gcdA = carl::gcd( cpConstantPart, cpFactor )*carl::gcd(ccConstantPart,ccFactor);
                }
            }
        }
        if( gcdA.isZero() ) return;
        Rational ccGcdA = gcdA.coprimeFactor();
        Poly cpGcdA = gcdA * ccGcdA;
        Rational ccDenominator = mDenominator.coprimeFactor();
        Poly cpDenominator = mDenominator * ccDenominator;
        gcdA = carl::gcd( cpGcdA, cpDenominator )*carl::gcd(ccGcdA,ccDenominator);
        // Make sure that the polynomial to divide by cannot be negative, otherwise the sign of the square root expression could change.
        if( !(gcdA == constant_one<Poly>::get()) && gcdA.definiteness() == carl::Definiteness::POSITIVE_SEMI )
        {
            if( !mConstantPart.isZero() )
            {
                mConstantPart.divideBy( gcdA, mConstantPart );
            }
            if( !mFactor.isZero() )
            {
                mFactor.divideBy( gcdA, mFactor );
            }
            mDenominator.divideBy( gcdA, mDenominator );
        }
        Rational numGcd = constant_zero<Rational>::get();
        Rational denomLcm = constant_one<Rational>::get();
        if( factor().isZero() )
        {
            if( !constantPart().isZero() )
            {
                Rational cpOfConstantPart = constantPart().coprimeFactor();
                numGcd = carl::getNum( cpOfConstantPart );
                denomLcm = carl::getDenom( cpOfConstantPart );
            }
            else
            {
//                cout << "        to " << *this << endl;
                return; // Sqrt expression corresponds to 0.
            }
        }
        else
        {
            Rational cpOfFactorPart = factor().coprimeFactor();
            if( constantPart().isZero() )
            {
                numGcd = carl::getNum( cpOfFactorPart );
                denomLcm = carl::getDenom( cpOfFactorPart );
            }
            else
            {
                Rational cpOfConstantPart = constantPart().coprimeFactor();
                numGcd = carl::gcd( carl::getNum( cpOfConstantPart ), carl::getNum( cpOfFactorPart ) );
                denomLcm = carl::lcm( carl::getDenom( cpOfConstantPart ), carl::getDenom( cpOfFactorPart ) );
            }
        }
        assert( numGcd != constant_zero<Rational>::get() );
        Rational cpFactor = numGcd/denomLcm; 
        mConstantPart *= cpFactor;
        mFactor *= cpFactor;
        Rational cpOfDenominator = denominator().coprimeFactor();
        mDenominator *= cpOfDenominator;
        Rational sqrtExFactor = (denomLcm*carl::getNum( cpOfDenominator ))/(numGcd*carl::getDenom( cpOfDenominator ));
        mConstantPart *= carl::getNum( sqrtExFactor );
        mFactor *= carl::getNum( sqrtExFactor );
        mDenominator *= carl::getDenom( sqrtExFactor );
//        cout << "       to  " << *this << endl;
        //TODO: implement this method further
    }

	template<typename Poly>
    bool SqrtEx<Poly>::operator==( const SqrtEx& _toCompareWith ) const
    {
        return    mRadicand == _toCompareWith.radicand() && mDenominator == _toCompareWith.denominator() 
               && mFactor == _toCompareWith.factor() && mConstantPart == _toCompareWith.constantPart();
    }

	template<typename Poly>
    SqrtEx<Poly>& SqrtEx<Poly>::operator=( const SqrtEx<Poly>& _sqrtEx )
    {
        if( this != &_sqrtEx )
        {
            mConstantPart = _sqrtEx.constantPart();
            mFactor       = _sqrtEx.factor();
            mDenominator  = _sqrtEx.denominator();
            if (factor().isZero())
                mRadicand = constant_zero<Poly>::get();
            else
                mRadicand = _sqrtEx.radicand();
        }
        return *this;
    }

	template<typename Poly>
    SqrtEx<Poly>& SqrtEx<Poly>::operator=( const Poly& _poly )
    {
        mConstantPart = _poly;
        mFactor       = constant_zero<Poly>::get();
        mDenominator  = constant_one<Poly>::get();
        mRadicand     = constant_zero<Poly>::get();
        return *this;
    }

	template<typename Poly>
    SqrtEx<Poly> SqrtEx<Poly>::operator+( const SqrtEx<Poly>& rhs ) const
    {
        assert( !hasSqrt() ||!rhs.hasSqrt() || radicand() == rhs.radicand() );
        SqrtEx<Poly> result = SqrtEx<Poly>( rhs.denominator() * constantPart() + rhs.constantPart() * denominator(),
                         rhs.denominator() * factor() + rhs.factor() * denominator(),
                         denominator() * rhs.denominator(), radicand() );
        return result;
    }

	template<typename Poly>
    SqrtEx<Poly> SqrtEx<Poly>::operator-( const SqrtEx<Poly>& rhs ) const
    {
        assert( !hasSqrt() || !rhs.hasSqrt() || radicand() == rhs.radicand() );
        SqrtEx<Poly> result = SqrtEx<Poly>( rhs.denominator() * constantPart() - rhs.constantPart() * denominator(),
                         rhs.denominator() * factor() - rhs.factor() * denominator(),
                         denominator() * rhs.denominator(), radicand() );
        return result;
    }

	template<typename Poly>
    SqrtEx<Poly> SqrtEx<Poly>::operator*( const SqrtEx<Poly>& rhs ) const
    {
        assert( !hasSqrt() || !rhs.hasSqrt() || radicand() == rhs.radicand() );
        SqrtEx<Poly> result = SqrtEx<Poly>( rhs.constantPart() * constantPart() + rhs.factor() * factor() * radicand(),
                         rhs.constantPart() * factor() + rhs.factor() * constantPart(),
                         denominator() * rhs.denominator(), radicand() );
        return result;
    }

	template<typename Poly>
    SqrtEx<Poly> SqrtEx<Poly>::operator/( const SqrtEx<Poly>& rhs ) const
    {
        assert( !rhs.hasSqrt() );
        SqrtEx<Poly> result = SqrtEx<Poly>( constantPart() * rhs.denominator(), factor() * rhs.denominator(),
                         denominator() * rhs.factor(), radicand() );
        return result;
    }
    
	template<typename Poly>
    inline std::ostream& operator<<( std::ostream& _out, const SqrtEx<Poly>& _substitution )
    {
        return (_out << _substitution.toString( true ) );
    }
    
	template<typename Poly>
    string SqrtEx<Poly>::toString( bool _infix, bool _friendlyNames ) const
    {
        if( _infix )
        {
            bool complexNum = hasSqrt() && !mConstantPart.isConstant();
            std::string result;
            if( complexNum && !mDenominator.isOne() )
                result += "(";
            if( hasSqrt() )
            {
                if( mConstantPart.isConstant() )
                    result += mConstantPart.toString( true, _friendlyNames );
                else
                    result += "(" + mConstantPart.toString( true, _friendlyNames ) + ")";
                result += "+";
                if( mFactor.isConstant() )
                    result += mFactor.toString( true, _friendlyNames );
                else
                    result += "(" + mFactor.toString( true, _friendlyNames ) + ")";
                result += "*sqrt(" + mRadicand.toString( true, _friendlyNames ) + ")";
            }
            else
            {
                if( mConstantPart.isConstant() || mDenominator.isOne())
                    result += mConstantPart.toString( true, _friendlyNames );
                else
                    result += "(" + mConstantPart.toString( true, _friendlyNames ) + ")";
            }
            if (!mDenominator.isOne())
            {
                if( complexNum )
                    result += ")";
                result += "/";
                if( mDenominator.isConstant() )
                    result += mDenominator.toString( true, _friendlyNames );
                else
                    result += "(" + mDenominator.toString( true, _friendlyNames ) + ")";
            }
            return result;
        }
        else
        {
            std::string result = "(/ (+ ";
            result += mConstantPart.toString( false, _friendlyNames );
            result +=  " (* ";
            result +=  mFactor.toString( false, _friendlyNames );
            result +=  " ";
            result +=  "(sqrt ";
            result +=  mRadicand.toString( false, _friendlyNames );
            result +=  "))) ";
            result +=  mDenominator.toString( false, _friendlyNames );
            result +=  ")";
            return result;
        }
    }
    
	template<typename Poly>
    bool SqrtEx<Poly>::evaluate(Rational& _result, const std::map<Variable, Rational>& _evalMap, int _rounding) const
    {
        Poly radicandEvaluated = radicand().substitute( _evalMap );
        assert( radicandEvaluated.isConstant() );
        Rational radicandValue = radicandEvaluated.constantPart();
        assert( radicandValue >= 0 );
        Poly factorEvaluated = factor().substitute( _evalMap );
        assert( factorEvaluated.isConstant() );
        Rational factorValue = factorEvaluated.constantPart();
        Poly constantPartEvaluated = constantPart().substitute( _evalMap );
        assert( constantPartEvaluated.isConstant() );
        Rational constantPartValue = constantPartEvaluated.constantPart();
        Poly denomEvaluated = denominator().substitute( _evalMap );
        assert( denomEvaluated.isConstant() );
        Rational denomValue = denomEvaluated.constantPart();
        assert( !carl::isZero( denomValue ) );
        // Check whether the resulting assignment is integer.
        bool rounded = true;
        Rational sqrtExValue;
        if( !carl::sqrt_exact( radicandValue, sqrtExValue ) )
        {
            assert( _rounding != 0 );
            rounded = false;
            assert( factorValue != 0 );
            double dbSqrt = sqrt( carl::toDouble( radicandValue ) );
            sqrtExValue = carl::rationalize<Rational>( dbSqrt ) ;
            // As there is no rational number representing the resulting square root we have to round.
            if( _rounding < 0 ) // If the result should round down in this case.
            {
                if( factorValue > 0 && (sqrtExValue*sqrtExValue) > radicandValue )
                {
                    // The factor of the resulting square root is positive, hence force rounding down.
                    dbSqrt = std::nextafter( dbSqrt, -INFINITY );
                    sqrtExValue = carl::rationalize<Rational>( dbSqrt );
                    assert( !((sqrtExValue*sqrtExValue) > radicandValue) );
                }
                else if( factorValue < 0 && (sqrtExValue*sqrtExValue) < radicandValue )
                {
                    // The factor of the resulting square root is negative, hence force rounding up.
                    dbSqrt = std::nextafter( dbSqrt, INFINITY );
                    sqrtExValue = carl::rationalize<Rational>( dbSqrt );
                    assert( !((sqrtExValue*sqrtExValue) < radicandValue) );
                }
            }
            else if( _rounding > 0 ) // If the result should round up in this case.
            {
                if( factorValue < 0 && (sqrtExValue*sqrtExValue) > radicandValue )
                {
                    // The factor of the resulting square root is negative, hence force rounding down.
                    dbSqrt = std::nextafter( dbSqrt, -INFINITY );
                    sqrtExValue = carl::rationalize<Rational>( dbSqrt );
                    assert( !((sqrtExValue*sqrtExValue) > radicandValue) );
                }
                else if( factorValue > 0 && (sqrtExValue*sqrtExValue) < radicandValue )
                {
                    // The factor of the resulting square root is positive, hence force rounding up.
                    dbSqrt = std::nextafter( dbSqrt, INFINITY );
                    sqrtExValue = carl::rationalize<Rational>( dbSqrt );
                    assert( !((sqrtExValue*sqrtExValue) < radicandValue) );
                }
            }
        }
        _result = (constantPartValue + factorValue * sqrtExValue) / denomValue;
        return rounded;
    }
    
	template<typename Poly>
    SqrtEx<Poly> SqrtEx<Poly>::substitute( const std::map<Variable, Rational>& _evalMap ) const
    {
        Poly radicandEvaluated = radicand().substitute( _evalMap );
        Poly factorEvaluated = factor().substitute( _evalMap );
        Poly constantPartEvaluated = constantPart().substitute( _evalMap );
        Poly denomEvaluated = denominator().substitute( _evalMap );
        assert( !denomEvaluated.isConstant() || !carl::isZero( denomEvaluated.constantPart() ) );
        Rational sqrtExValue;
        if( radicandEvaluated.isConstant() && carl::sqrt_exact( radicandEvaluated.constantPart(), sqrtExValue ) )
        {
            return SqrtEx<Poly>(Poly(constantPartEvaluated + factorEvaluated * sqrtExValue), 
                    constant_zero<Poly>::get(), 
                    std::move(denomEvaluated), 
                    constant_zero<Poly>::get());
        }
        return SqrtEx( std::move(constantPartEvaluated), std::move(factorEvaluated), std::move(denomEvaluated), std::move(radicandEvaluated) );
    }

	template<typename Poly>
    SqrtEx<Poly> SqrtEx<Poly>::subBySqrtEx( const Poly& _substituteIn, const carl::Variable& _varToSubstitute, const SqrtEx<Poly>& _substituteBy )
    {
        if( !_substituteIn.has( _varToSubstitute ) )
            return SqrtEx<Poly>( _substituteIn );
        /*
         * We have to calculate the result of the substitution:
         *
         *                           q+r*sqrt{t}
         *        (a_n*x^n+...+a_0)[------------ / x]
         *                               s
         * being:
         *
         *      sum_{k=0}^n (a_k * (q+r*sqrt{t})^k * s^{n-k})
         *      ----------------------------------------------
         *                           s^n
         */
        auto varInfo = _substituteIn.template getVarInfo<true>( _varToSubstitute );
        const auto& coeffs = varInfo.coeffs();
        // Calculate the s^k:   (0<=k<=n)
        auto coeff = coeffs.begin();
        carl::uint lastDegree = varInfo.maxDegree();
        std::vector<Poly> sk;
        sk.push_back(constant_one<Poly>::get());
        for( carl::uint i = 1; i <= lastDegree; ++i )
        {
            // s^i = s^l * s^{i-l}
            sk.push_back( sk.back() * _substituteBy.denominator() );
        }
        // Calculate the constant part and factor of the square root of (q+r*sqrt{t})^k 
        std::vector<Poly> qk;
        qk.push_back( _substituteBy.constantPart() );
        std::vector<Poly> rk;
        rk.push_back( _substituteBy.factor() );
        // Let (q+r*sqrt{t})^l be (q'+r'*sqrt{t}) 
        // then (q+r*sqrt{t})^l+1  =  (q'+r'*sqrt{t}) * (q+r*sqrt{t})  =  ( q'*q+r'*r't  +  (q'*r+r'*q) * sqrt{t} )
        for( carl::uint i = 1; i < lastDegree; ++i )
        {
            // q'*q+r'*r't
            qk.push_back( qk.back() * _substituteBy.constantPart() + rk.back() * _substituteBy.factor() * _substituteBy.radicand() );
            // q'*r+r'*q
            rk.push_back( rk.back() * _substituteBy.constantPart()  + qk.at( i - 1 ) * _substituteBy.factor() );
        }
        // Calculate the result:
        Poly resFactor = constant_zero<Poly>::get();
        Poly resConstantPart = constant_zero<Poly>::get();
        if( coeff->first == 0 )
        {
            resConstantPart += sk.back() * coeff->second;
            ++coeff;
        }
        for( ; coeff != coeffs.end(); ++coeff )
        {
            resConstantPart += coeff->second * qk.at( coeff->first - 1 ) * sk.at( lastDegree - coeff->first );
            resFactor       += coeff->second * rk.at( coeff->first - 1 ) * sk.at( lastDegree - coeff->first );
        }
        return SqrtEx( resConstantPart, resFactor, sk.back(), _substituteBy.radicand() );
    }
}    // end namspace vs
