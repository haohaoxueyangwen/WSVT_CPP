#include "umpa_fixture_data.hpp"
#include "wsvt/umpa_physical_fit.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

void emit_number(double value) {
    if (std::isfinite(value)) {
        std::cout << value;
    } else {
        std::cout << "null";
    }
}

void emit_fit(
    const std::string& id, const wsvt::UmpaPhysicalFit& fit) {
    std::cout << "{\"id\":\"" << id << "\",\"l1\":";
    emit_number(fit.statistics.l1);
    std::cout << ",\"l2\":";
    emit_number(fit.statistics.l2);
    std::cout << ",\"l3\":";
    emit_number(fit.statistics.l3);
    std::cout << ",\"l4\":";
    emit_number(fit.statistics.l4);
    std::cout << ",\"l5\":";
    emit_number(fit.statistics.l5);
    std::cout << ",\"l6\":";
    emit_number(fit.statistics.l6);
    std::cout << ",\"weight_sum\":";
    emit_number(fit.statistics.weight_sum);
    std::cout << ",\"observation_count\":" << fit.statistics.observation_count;
    std::cout << ",\"alpha\":";
    emit_number(fit.alpha);
    std::cout << ",\"beta\":";
    emit_number(fit.beta);
    std::cout << ",\"transmission\":";
    emit_number(fit.transmission);
    std::cout << ",\"visibility\":";
    emit_number(fit.visibility);
    std::cout << ",\"cost\":";
    emit_number(fit.cost);
    std::cout << ",\"delta\":";
    emit_number(fit.delta);
    std::cout << ",\"condition\":";
    emit_number(fit.condition);
    std::cout << ",\"numerical_valid\":"
              << (fit.numerical_valid ? "true" : "false")
              << ",\"physical_valid\":"
              << (fit.physical_valid ? "true" : "false") << '}';
}

}  // namespace

int main() {
    std::cout << std::setprecision(17)
              << "{\"schema_version\":1,\"implementation\":"
                 "\"cpp_scalar_float64\",\"analysis_radius\":2,\"cases\":[";
    bool first = true;
    for (const auto& fixture : umpa_fixture::all_fixtures()) {
        const wsvt::UmpaPhysicalFit fit = wsvt::fit_umpa_physical(
            fixture.sample, fixture.reference, fixture.reference_mean, fixture.weights);
        if (!first) {
            std::cout << ',';
        }
        first = false;
        emit_fit(fixture.id, fit);
    }
    std::cout << "]}\n";
    return 0;
}
