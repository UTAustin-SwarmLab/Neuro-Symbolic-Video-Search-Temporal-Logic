#pragma once

#include <boost/optional.hpp>
#include <map>
#include <string>
#include <vector>

namespace storm {
namespace prism {
class Program;
}
namespace jani {
class Model;
class ModelFeatures;
class Property;
}  // namespace jani

namespace api {

storm::prism::Program parseProgram(std::string const& filename, bool prismCompatibility = false, bool simplify = true);

std::pair<storm::jani::Model, std::vector<storm::jani::Property>> parseJaniModel(std::string const& filename,
                                                                                 boost::optional<std::vector<std::string>> const& propertyFilter = boost::none);
std::pair<storm::jani::Model, std::vector<storm::jani::Property>> parseJaniModel(std::string const& filename, storm::jani::ModelFeatures const& allowedFeatures,
                                                                                 boost::optional<std::vector<std::string>> const& propertyFilter = boost::none);
std::pair<storm::jani::Model, std::vector<storm::jani::Property>> parseJaniModelFromString(
    std::string const& jsonstring, boost::optional<std::vector<std::string>> const& propertyFilter = boost::none);
std::pair<storm::jani::Model, std::vector<storm::jani::Property>> parseJaniModelFromString(
    std::string const& jsonstring, storm::jani::ModelFeatures const& allowedFeatures,
    boost::optional<std::vector<std::string>> const& propertyFilter = boost::none);
void simplifyJaniModel(storm::jani::Model& model, std::vector<storm::jani::Property>& properties, storm::jani::ModelFeatures const& supportedFeatures);

}  // namespace api
}  // namespace storm
