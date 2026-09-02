#include "wsvt/io_json.hpp"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace wsvt {

JsonValue::JsonValue() : value(nullptr) {}
JsonValue::JsonValue(std::nullptr_t v) : value(v) {}
JsonValue::JsonValue(bool v) : value(v) {}
JsonValue::JsonValue(double v) : value(v) {}
JsonValue::JsonValue(int v) : value(static_cast<double>(v)) {}
JsonValue::JsonValue(const std::string& v) : value(v) {}
JsonValue::JsonValue(const char* v) : value(std::string(v)) {}
JsonValue::JsonValue(const JsonArray& v) : value(v) {}
JsonValue::JsonValue(const JsonObject& v) : value(v) {}

namespace {

std::string escape_json_string(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (const char c : s) {
        switch (c) {
            case '\"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out.push_back(c); break;
        }
    }
    return out;
}

void dumps_impl(const JsonValue& v, std::ostringstream& os) {
    if (std::holds_alternative<std::nullptr_t>(v.value)) {
        os << "null";
        return;
    }
    if (std::holds_alternative<bool>(v.value)) {
        os << (std::get<bool>(v.value) ? "true" : "false");
        return;
    }
    if (std::holds_alternative<double>(v.value)) {
        os << std::get<double>(v.value);
        return;
    }
    if (std::holds_alternative<std::string>(v.value)) {
        os << "\"" << escape_json_string(std::get<std::string>(v.value)) << "\"";
        return;
    }
    if (std::holds_alternative<JsonArray>(v.value)) {
        const auto& arr = std::get<JsonArray>(v.value);
        os << "[";
        for (std::size_t i = 0; i < arr.size(); ++i) {
            dumps_impl(arr[i], os);
            if (i + 1 < arr.size()) {
                os << ",";
            }
        }
        os << "]";
        return;
    }
    const auto& obj = std::get<JsonObject>(v.value);
    os << "{";
    std::size_t i = 0;
    for (const auto& kv : obj) {
        os << "\"" << escape_json_string(kv.first) << "\":";
        dumps_impl(kv.second, os);
        if (i + 1 < obj.size()) {
            os << ",";
        }
        ++i;
    }
    os << "}";
}

class JsonParser {
public:
    explicit JsonParser(std::string src) : src_(std::move(src)), pos_(0) {}

    JsonValue parse_value() {
        skip_ws();
        if (pos_ >= src_.size()) {
            throw std::runtime_error("json parse error: unexpected eof");
        }
        const char c = src_[pos_];
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return JsonValue(parse_string());
        if (c == 't') return parse_true();
        if (c == 'f') return parse_false();
        if (c == 'n') return parse_null();
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parse_number();
        throw std::runtime_error("json parse error: invalid token");
    }

private:
    std::string src_;
    std::size_t pos_;

    void skip_ws() {
        while (pos_ < src_.size() && std::isspace(static_cast<unsigned char>(src_[pos_]))) {
            ++pos_;
        }
    }

    void expect(char ch) {
        if (pos_ >= src_.size() || src_[pos_] != ch) {
            throw std::runtime_error("json parse error: expected char");
        }
        ++pos_;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (pos_ < src_.size()) {
            const char c = src_[pos_++];
            if (c == '"') {
                return out;
            }
            if (c == '\\') {
                if (pos_ >= src_.size()) {
                    throw std::runtime_error("json parse error: bad escape");
                }
                const char e = src_[pos_++];
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    default: throw std::runtime_error("json parse error: unsupported escape");
                }
            } else {
                out.push_back(c);
            }
        }
        throw std::runtime_error("json parse error: unterminated string");
    }

    JsonValue parse_true() {
        if (src_.substr(pos_, 4) != "true") {
            throw std::runtime_error("json parse error: invalid true");
        }
        pos_ += 4;
        return JsonValue(true);
    }

    JsonValue parse_false() {
        if (src_.substr(pos_, 5) != "false") {
            throw std::runtime_error("json parse error: invalid false");
        }
        pos_ += 5;
        return JsonValue(false);
    }

    JsonValue parse_null() {
        if (src_.substr(pos_, 4) != "null") {
            throw std::runtime_error("json parse error: invalid null");
        }
        pos_ += 4;
        return JsonValue(nullptr);
    }

    JsonValue parse_number() {
        const std::size_t start = pos_;
        if (src_[pos_] == '-') {
            ++pos_;
        }
        while (pos_ < src_.size() && std::isdigit(static_cast<unsigned char>(src_[pos_]))) {
            ++pos_;
        }
        if (pos_ < src_.size() && src_[pos_] == '.') {
            ++pos_;
            while (pos_ < src_.size() && std::isdigit(static_cast<unsigned char>(src_[pos_]))) {
                ++pos_;
            }
        }
        if (pos_ < src_.size() && (src_[pos_] == 'e' || src_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < src_.size() && (src_[pos_] == '+' || src_[pos_] == '-')) {
                ++pos_;
            }
            while (pos_ < src_.size() && std::isdigit(static_cast<unsigned char>(src_[pos_]))) {
                ++pos_;
            }
        }
        const double v = std::stod(src_.substr(start, pos_ - start));
        return JsonValue(v);
    }

    JsonValue parse_array() {
        expect('[');
        skip_ws();
        JsonArray arr;
        if (pos_ < src_.size() && src_[pos_] == ']') {
            ++pos_;
            return JsonValue(arr);
        }
        while (true) {
            arr.push_back(parse_value());
            skip_ws();
            if (pos_ >= src_.size()) {
                throw std::runtime_error("json parse error: unterminated array");
            }
            if (src_[pos_] == ']') {
                ++pos_;
                return JsonValue(arr);
            }
            expect(',');
        }
    }

    JsonValue parse_object() {
        expect('{');
        skip_ws();
        JsonObject obj;
        if (pos_ < src_.size() && src_[pos_] == '}') {
            ++pos_;
            return JsonValue(obj);
        }
        while (true) {
            skip_ws();
            const std::string key = parse_string();
            skip_ws();
            expect(':');
            JsonValue val = parse_value();
            obj[key] = std::move(val);
            skip_ws();
            if (pos_ >= src_.size()) {
                throw std::runtime_error("json parse error: unterminated object");
            }
            if (src_[pos_] == '}') {
                ++pos_;
                return JsonValue(obj);
            }
            expect(',');
        }
    }
};

}

std::string json_dumps(const JsonValue& v, int indent) {
    (void)indent;
    std::ostringstream os;
    os << std::setprecision(std::numeric_limits<double>::max_digits10);
    dumps_impl(v, os);
    return os.str();
}

void write_json(
    const std::string& result_path,
    const std::string& file_name,
    const JsonObject& data_dict) {
    std::filesystem::path out_dir(result_path);
    if (!std::filesystem::exists(out_dir)) {
        std::filesystem::create_directories(out_dir);
    }
    const std::filesystem::path file_path = out_dir / (file_name + ".json");
    std::ofstream fp(file_path, std::ios::out | std::ios::trunc);
    if (!fp) {
        throw std::runtime_error("failed to open json file for write");
    }
    fp << json_dumps(JsonValue(data_dict), 0);
    fp.close();
    std::cout << "result json file : " << file_name << ".json saved" << std::endl;
}

JsonObject read_json(
    const std::string& filepath,
    bool print_para) {
    if (!std::filesystem::exists(filepath)) {
        throw std::runtime_error("Wrong file path");
    }
    std::ifstream fp(filepath, std::ios::in);
    if (!fp) {
        throw std::runtime_error("failed to open json file for read");
    }
    std::ostringstream ss;
    ss << fp.rdbuf();
    fp.close();

    JsonParser parser(ss.str());
    JsonValue root = parser.parse_value();
    if (!std::holds_alternative<JsonObject>(root.value)) {
        throw std::runtime_error("json root is not object");
    }
    JsonObject data = std::get<JsonObject>(root.value);
    if (print_para) {
        std::cout << "parameters: " << json_dumps(JsonValue(data), 0) << std::endl;
    }
    return data;
}

}
