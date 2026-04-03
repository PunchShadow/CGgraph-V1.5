#pragma once

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace gflags {
namespace compat {

enum class FlagType { Int32, Int64, String };

struct FlagRecord
{
    FlagType type;
    std::string help;
    std::string default_value;
    void* storage;
};

inline std::unordered_map<std::string, FlagRecord>& registry()
{
    static std::unordered_map<std::string, FlagRecord> records;
    return records;
}

inline std::vector<std::string>& registration_order()
{
    static std::vector<std::string> names;
    return names;
}

inline std::string& usage_message()
{
    static std::string message;
    return message;
}

inline bool register_int32(const char* name, int32_t* storage, int32_t default_value, const char* help)
{
    *storage = default_value;
    registry()[name] = {FlagType::Int32, help ? help : "", std::to_string(default_value), storage};
    registration_order().emplace_back(name);
    return true;
}

inline bool register_int64(const char* name, int64_t* storage, int64_t default_value, const char* help)
{
    *storage = default_value;
    registry()[name] = {FlagType::Int64, help ? help : "", std::to_string(default_value), storage};
    registration_order().emplace_back(name);
    return true;
}

inline bool register_string(const char* name, std::string* storage, const std::string& default_value, const char* help)
{
    *storage = default_value;
    registry()[name] = {FlagType::String, help ? help : "", default_value, storage};
    registration_order().emplace_back(name);
    return true;
}

inline bool parse_int32(const std::string& value, int32_t* out)
{
    try
    {
        const long long parsed = std::stoll(value);
        if (parsed < std::numeric_limits<int32_t>::min() || parsed > std::numeric_limits<int32_t>::max()) return false;
        *out = static_cast<int32_t>(parsed);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

inline bool parse_int64(const std::string& value, int64_t* out)
{
    try
    {
        *out = std::stoll(value);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

inline bool set_flag(const std::string& name, const std::string& value, std::string* error_message)
{
    auto it = registry().find(name);
    if (it == registry().end())
    {
        if (error_message) *error_message = "unknown flag";
        return false;
    }

    FlagRecord& record = it->second;
    switch (record.type)
    {
    case FlagType::Int32: {
        int32_t parsed = 0;
        if (!parse_int32(value, &parsed))
        {
            if (error_message) *error_message = "expect int32";
            return false;
        }
        *static_cast<int32_t*>(record.storage) = parsed;
        return true;
    }
    case FlagType::Int64: {
        int64_t parsed = 0;
        if (!parse_int64(value, &parsed))
        {
            if (error_message) *error_message = "expect int64";
            return false;
        }
        *static_cast<int64_t*>(record.storage) = parsed;
        return true;
    }
    case FlagType::String:
        *static_cast<std::string*>(record.storage) = value;
        return true;
    }

    if (error_message) *error_message = "invalid flag type";
    return false;
}

} // namespace compat

inline void SetUsageMessage(const std::string& message) { compat::usage_message() = message; }

inline void ShowUsageWithFlags()
{
    if (!compat::usage_message().empty()) std::cout << compat::usage_message() << '\n';
    std::cout << "Flags:\n";
    for (const std::string& name : compat::registration_order())
    {
        const auto& record = compat::registry().at(name);
        std::cout << "  -" << name << " (default: " << record.default_value << ")\n"
                  << "      " << record.help << '\n';
    }
}

inline void ParseCommandLineFlags(int* argc, char*** argv, bool remove_flags)
{
    if (!argc || !argv || !(*argv) || *argc <= 0) return;

    std::vector<char*> kept_args;
    kept_args.reserve(static_cast<size_t>(*argc) + 1);
    kept_args.push_back((*argv)[0]);

    for (int i = 1; i < *argc; ++i)
    {
        const std::string arg = ((*argv)[i] != nullptr) ? (*argv)[i] : "";

        if (arg == "--")
        {
            for (int j = i + 1; j < *argc; ++j) kept_args.push_back((*argv)[j]);
            break;
        }

        if (arg == "--help" || arg == "-help" || arg == "-h" || arg == "--helpshort")
        {
            ShowUsageWithFlags();
            std::exit(0);
        }

        if (arg.size() > 1 && arg[0] == '-')
        {
            const std::string token = (arg.rfind("--", 0) == 0) ? arg.substr(2) : arg.substr(1);
            if (token.empty())
            {
                kept_args.push_back((*argv)[i]);
                continue;
            }

            const size_t eq_pos = token.find('=');
            const std::string flag_name = (eq_pos == std::string::npos) ? token : token.substr(0, eq_pos);
            const auto flag_it = compat::registry().find(flag_name);

            if (flag_it == compat::registry().end())
            {
                kept_args.push_back((*argv)[i]);
                continue;
            }

            std::string flag_value;
            bool consume_next = false;
            if (eq_pos != std::string::npos)
            {
                flag_value = token.substr(eq_pos + 1);
            }
            else
            {
                if (i + 1 >= *argc)
                {
                    std::cerr << "Missing value for flag -" << flag_name << '\n';
                    std::exit(1);
                }
                flag_value = ((*argv)[i + 1] != nullptr) ? (*argv)[i + 1] : "";
                consume_next = true;
            }

            std::string error;
            if (!compat::set_flag(flag_name, flag_value, &error))
            {
                std::cerr << "Failed to parse flag -" << flag_name << ": " << error << '\n';
                std::exit(1);
            }

            if (!remove_flags)
            {
                kept_args.push_back((*argv)[i]);
                if (consume_next) kept_args.push_back((*argv)[i + 1]);
            }

            if (consume_next) ++i;
            continue;
        }

        kept_args.push_back((*argv)[i]);
    }

    if (!remove_flags) return;

    const int new_argc = static_cast<int>(kept_args.size());
    for (int i = 0; i < new_argc; ++i) (*argv)[i] = kept_args[static_cast<size_t>(i)];
    (*argv)[new_argc] = nullptr;
    *argc = new_argc;
}

} // namespace gflags

#define DECLARE_string(name) \
    namespace fLS { extern std::string FLAGS_##name; } \
    using fLS::FLAGS_##name

#define DECLARE_int32(name) \
    namespace fLI { extern int32_t FLAGS_##name; } \
    using fLI::FLAGS_##name

#define DECLARE_int64(name) \
    namespace fLI64 { extern int64_t FLAGS_##name; } \
    using fLI64::FLAGS_##name

#define DEFINE_string(name, default_value, help_text)                                                                                             \
    namespace fLS { std::string FLAGS_##name = (default_value); }                                                                                 \
    using fLS::FLAGS_##name;                                                                                                                       \
    namespace                                                                                                                                    \
    {                                                                                                                                           \
    const bool FLAGS_register_##name = ::gflags::compat::register_string(#name, &fLS::FLAGS_##name, (default_value), (help_text));            \
    }

#define DEFINE_int32(name, default_value, help_text)                                                                                              \
    namespace fLI { int32_t FLAGS_##name = static_cast<int32_t>(default_value); }                                                                \
    using fLI::FLAGS_##name;                                                                                                                       \
    namespace                                                                                                                                    \
    {                                                                                                                                           \
    const bool FLAGS_register_##name =                                                                                                            \
        ::gflags::compat::register_int32(#name, &fLI::FLAGS_##name, static_cast<int32_t>(default_value), (help_text));                         \
    }

#define DEFINE_int64(name, default_value, help_text)                                                                                              \
    namespace fLI64 { int64_t FLAGS_##name = static_cast<int64_t>(default_value); }                                                              \
    using fLI64::FLAGS_##name;                                                                                                                     \
    namespace                                                                                                                                    \
    {                                                                                                                                           \
    const bool FLAGS_register_##name =                                                                                                            \
        ::gflags::compat::register_int64(#name, &fLI64::FLAGS_##name, static_cast<int64_t>(default_value), (help_text));                       \
    }
