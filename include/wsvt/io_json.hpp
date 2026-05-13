#pragma once

#include <map>
#include <string>
#include <variant>
#include <vector>

namespace wsvt {

struct JsonValue;
using JsonObject = std::map<std::string, JsonValue>;
using JsonArray = std::vector<JsonValue>;

struct JsonValue {
    using Variant = std::variant<std::nullptr_t, bool, double, std::string, JsonArray, JsonObject>;
    Variant value;
    JsonValue();
    JsonValue(std::nullptr_t v);
    JsonValue(bool v);
    JsonValue(double v);
    JsonValue(int v);
    JsonValue(const std::string& v);
    JsonValue(const char* v);
    JsonValue(const JsonArray& v);
    JsonValue(const JsonObject& v);
};

void write_json(
    const std::string& result_path,
    const std::string& file_name,
    const JsonObject& data_dict);

JsonObject read_json(
    const std::string& filepath,
    bool print_para = false);

std::string json_dumps(const JsonValue& v, int indent = 0);

}
