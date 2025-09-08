#ifndef STORM_SETTINGS_MODULES_CUDDSETTINGS_H_
#define STORM_SETTINGS_MODULES_CUDDSETTINGS_H_

#include "storm/settings/modules/ModuleSettings.h"

namespace storm {
namespace settings {
namespace modules {

/*!
 * This class represents the settings for CUDD.
 */
class CuddSettings : public ModuleSettings {
   public:
    // An enumeration of all available reordering techniques of CUDD.
    enum class ReorderingTechnique {
        None,
        Random,
        RandomPivot,
        Sift,
        SiftConv,
        SymmetricSift,
        SymmetricSiftConv,
        GroupSift,
        GroupSiftConv,
        Win2,
        Win2Conv,
        Win3,
        Win3Conv,
        Win4,
        Win4Conv,
        Annealing,
        Genetic,
        Exact
    };

    /*!
     * Creates a new set of CUDD settings.
     */
    CuddSettings();

    /*!
     * Retrieves the precision that CUDD is supposed to use for distinguishing constants.
     *
     * @return The desired precision of CUDD.
     */
    double getConstantPrecision() const;

    /*!
     * Retrieves the maximal amount of memory (in megabytes) that CUDD can occupy.
     *
     * @return The maximal amount of memory to use.
     */
    uint_fast64_t getMaximalMemory() const;

    /*!
     * Retrieves whether dynamic reordering is enabled.
     *
     * @return True iff dynamic reordering is enabled.
     */
    bool isReorderingEnabled() const;

    /*!
     * Retrieves the reordering technique that CUDD is supposed to use.
     *
     * @return The reordering technique to use.
     */
    ReorderingTechnique getReorderingTechnique() const;

    // The name of the module.
    static const std::string moduleName;

   private:
    // Define the string names of the options as constants.
    static const std::string precisionOptionName;
    static const std::string maximalMemoryOptionName;
    static const std::string reorderOptionName;
    static const std::string reorderTechniqueOptionName;
};

}  // namespace modules
}  // namespace settings
}  // namespace storm

#endif /* STORM_SETTINGS_MODULES_CUDDSETTINGS_H_ */
