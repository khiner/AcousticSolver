#include "DgReference.h"
#include <algorithm>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <mach-o/dyld.h>
#include <spawn.h>
#include <sstream>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;
namespace dg::reference {
namespace {
fs::path Nodes(const Fixture &f) {
    std::ostringstream name;
    name << "tetN" << std::setw(2) << std::setfill('0') << f.Degree << ".dat";
    return FixtureRoot / "nodes" / name.str();
}
fs::path Operator(const fs::path &inputs, const Fixture &f, bool reduced = true) {
    return inputs / (reduced ? "reduced" : "factored") / f.Name / (reduced ? "q" + std::to_string(f.Order) : "operator");
}
fs::path Executable() {
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::vector<char> path(size);
    Require(_NSGetExecutablePath(path.data(), &size) == 0, "Cannot locate reference executable");
    return fs::canonical(path.data());
}
Json Runtime() {
    return {{"executable_sha256", Hash(Executable())}, {"compiler", __VERSION__}, {"eigen_version", {EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION}}, {"blas", "Accelerate"}, {"threads", 1}};
}
Json Sources() {
    Json hashes = Json::object();
    for (const auto &entry : fs::directory_iterator(Project / "src/Dg"))
        if (entry.is_regular_file()) hashes[fs::relative(entry.path(), Project).string()] = Hash(entry.path());
    return hashes;
}
void Gate(const Json &errors, double tolerance) {
    for (const auto &value : errors) Require(value.is_number() && std::isfinite(double(value)) && double(value) < tolerance, "Precision gate failed: " + errors.dump());
}
void Accurate(const Json &score) { Require(double(score.at("terminal_relative_acoustic_l2")) < 1e-3, "Analytic accuracy gate failed: " + score.dump()); }
int Run(const std::vector<std::string> &command, const fs::path &log) {
    std::vector<char *> argv;
    argv.reserve(command.size());
    for (const auto &a : command) argv.push_back(const_cast<char *>(a.c_str()));
    argv.push_back(nullptr);
    posix_spawn_file_actions_t actions;
    Require(posix_spawn_file_actions_init(&actions) == 0, "Cannot initialize process actions");
    int setup = posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (!setup) setup = posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);
    pid_t pid = 0;
    int const error = setup ? setup : posix_spawnp(&pid, argv[0], &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    Require(!error, "Cannot start " + command[0] + ": " + std::to_string(error));
    int status;
    while (waitpid(pid, &status, 0) < 0) Require(errno == EINTR, "Cannot wait for child process");
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128;
}
std::string Contents(const fs::path &file) {
    std::ifstream in(file);
    return {std::istreambuf_iterator<char>(in), {}};
}
int Thermal(const fs::path &log) {
    Require(Run({"osascript", "-l", "JavaScript", "-e", "ObjC.import(\"Foundation\"); $.NSProcessInfo.processInfo.thermalState"}, log) == 0, "Cannot read thermal state");
    int const state = std::stoi(Contents(log));
    Require(state >= 0 && state <= 3, "Invalid thermal state");
    return state;
}
double Median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return .5 * (v[(v.size() - 1) / 2] + v[v.size() / 2]);
}
void CopyOperator(const fs::path &source, const fs::path &dest, int copies = 1, bool tail = false) {
    NewDirectory(dest);
    fs::copy(source, dest, fs::copy_options::recursive);
    Json meta = Load(source / "operator.json");
    int e = meta.at("elements"), n = meta.at("nodes");
    for (const auto &[name, item] : meta["files"].items()) {
        std::vector<int> shape = item.at("shape");
        std::string const type = item.at("dtype");
        auto const transform = [&]<class T>() {
            auto original = Read<T>(source / (name + ".bin")), data = original;
            int parts = 1;
            if (shape[0] == e) {
                if (tail) data.resize(data.size() / e * (e - 1));
                else parts = copies;
                shape[0] = tail ? e - 1 : e * copies;
            } else if (name.starts_with("receiver_") && copies > 1) {
                parts = copies;
                shape[0] *= copies;
            }
            if (parts > 1) {
                data.resize(original.size() * parts);
                for (int c = 0; c < parts; ++c) {
                    int offset = 0;
                    if (name == "vmapM" || name == "vmapP") offset = c * e * n;
                    if (name == "receiver_elements") offset = c * e;
                    for (size_t i = 0; i < original.size(); ++i) data[c * original.size() + i] = original[i] + T(offset);
                }
            }
            if (tail) {
                if (name == "initial" || name == "initial_state" || name == "random_state" || name == "constant_state") {
                    std::fill(data.begin(), data.end(), T(0));
                    for (size_t i = 0; i < data.size(); i += 4) data[i] = 1;
                }
                if (name.ends_with("_rhs") || name == "receiver_elements") std::fill(data.begin(), data.end(), T(0));
                if (name == "boundary") std::fill(data.begin(), data.end(), T(1));
                if (name == "vmapP") {
                    data = Read<T>(source / "vmapM.bin");
                    data.resize(data.size() / e * (e - 1));
                }
            }
            Write(dest / (name + ".bin"), data);
        };
        if (type == "float32") transform.operator()<float>();
        else if (type == "float64") transform.operator()<double>();
        else {
            Require(type == "int32", "Unknown operator type");
            transform.operator()<int32_t>();
        }
        item["shape"] = shape;
        item["sha256"] = Hash(dest / (name + ".bin"));
    }
    meta["elements"] = tail ? e - 1 : e * copies;
    meta["receivers"] = int(meta["receivers"]) * copies;
    Save(dest / "operator.json", meta);
}
std::vector<int32_t> Indices(const fs::path &path, int bytes) {
    if (bytes == 4) return Read<int32_t>(path);
    Require(bytes == 8, "Unsupported upstream index width");
    auto const data = Read<int64_t>(path);
    std::vector<int32_t> out;
    for (auto v : data) {
        Require(v >= INT32_MIN && v <= INT32_MAX, "Upstream index exceeds int32");
        out.push_back(int32_t(v));
    }
    return out;
}
void Regenerate(const std::string &name, const fs::path &output, bool wadg) {
    VerifyFixtures();
    const auto *found = std::find_if(Fixtures.begin(), Fixtures.end(), [&](const Fixture &f) { return name == f.Name; });
    Require(found != Fixtures.end(), "Unknown reference case");
    const auto &f = *found;
    NewDirectory(output);
    Reference ref(FixtureRoot / "meshes" / f.MeshName, Nodes(f), wadg);
    Save(output / "operator-check.json", ref.Check());
    // Only the norm tables are needed to compare dense trajectories, including full-mass runs.
    auto const tables = output / "norm";
    NewDirectory(tables);
    Save(tables / "operator.json", {{"elements", ref.Source.E}, {"nodes", ref.Source.N}, {"pressure_offset", 2}, {"impedance", ref.Source.Z}});
    Mat h(ref.Source.E * ref.Source.N, ref.Source.N);
    for (int e = 0; e < ref.Source.E; ++e) h.middleRows(e * ref.Source.N, ref.Source.N) = ref.Elements[e].H;
    Write<double>(tables / "modified_mass.bin", {h.data(), size_t(h.size())});
    for (int repeat = 1; repeat <= 2; ++repeat) {
        auto const record = output / ("repeat" + std::to_string(repeat));
        ref.Run(record);
        auto const score = Score(ref.Source.Directory, record, Nodes(f));
        Accurate(score);
        Save(record / "analytic.json", score);
        if (wadg) {
            auto const errors = Compare(FixtureRoot / "cpu" / f.MeshName / "wadg/repeat1", record, tables);
            Gate(errors, 1e-10);
            Save(record / "versus-frozen.json", errors);
        }
    }
    for (const auto &file : {"final_q.bin", "receivers.bin", "times.bin"}) Require(Read<unsigned char>(output / "repeat1" / file) == Read<unsigned char>(output / "repeat2" / file), "CPU repeats differ");
    Save(output / "provenance.json", {{"form", wadg ? "wadg" : "full_mass"}, {"precision", "float64"}, {"device", "cpu"}, {"repeatable_bytes", true}, {"fixture_manifest_sha256", Hash(FixtureRoot / "sha256.json")}, {"source_sha256", Sources()}, {"runtime", Runtime()}});
}
} // namespace
void VerifyFixtures() {
    const auto manifest = Load(FixtureRoot / "sha256.json");
    for (const auto &[name, digest] : manifest.items()) Require(Hash(FixtureRoot / name) == digest.get<std::string>(), "DG fixture checksum mismatch: " + name);
}
void Prepare(const fs::path &output) {
    VerifyFixtures();
    fs::create_directories(output);
    fs::remove(output / "preparation.json");
    for (const auto &f : Fixtures) {
        std::cout << "Preparing " << f.Name << std::endl;
        Reference const ref(FixtureRoot / "meshes" / f.MeshName, Nodes(f));
        auto const full = Operator(output, f, false);
        ref.Export(full);
        Save(full / "reference-check.json", ref.Check());
        ref.Export(Operator(output, f), 2, f.Order);
    }
    Save(output / "preparation.json", {{"fixture_manifest_sha256", Hash(FixtureRoot / "sha256.json")}, {"source_sha256", Sources()}, {"runtime", Runtime()}});
}
void ImportUpstream(const fs::path &captures, const fs::path &checkout, const fs::path &output) {
    NewDirectory(output);
    for (const std::string case_name : {"cube_rigid", "cylinder"}) {
        auto capture = captures / "cuda" / case_name, a_dir = capture / "fp64/mesh", b_dir = capture / "fp32/mesh", dest = output / case_name, source = dest / "input";
        NewDirectory(source);
        auto a = Load(a_dir / "solver.json"), b = Load(b_dir / "solver.json");
        for (const auto &key : {"elements", "degree", "nodes_per_element", "nodes_per_face", "receivers"}) Require(a.at(key) == b.at(key), "Upstream precisions have different meshes");
        Require(a.at("scalar_bytes") == 8 && b.at("scalar_bytes") == 4, "Unexpected upstream precision");
        auto cfg = Load(a_dir / "provenance.json").at("config");
        a["initial_condition"] = "captured";
        a["density"] = std::stod(cfg.at("RHO").get<std::string>());
        a["sound_speed"] = std::stod(cfg.at("C").get<std::string>());
        Mat xyz(int(a.at("elements")) * int(a.at("nodes_per_element")), 3);
        for (int axis = 0; axis < 3; ++axis) xyz.col(axis) = ReadMatrix(a_dir / (std::string(1, "xyz"[axis]) + ".bin"), xyz.rows(), 1);
        Write<double>(source / "xyz.bin", {xyz.data(), size_t(xyz.size())});
        for (const auto &name : {"initial_q", "receiver_interpolation"}) {
            auto values = Read<float>(b_dir / (std::string(name) + ".bin"));
            Write(source / (std::string(name) + ".bin"), std::vector<double>(values.begin(), values.end()));
        }
        std::vector<int32_t> local_map, tags;
        for (const auto &name : {"vmapM", "vmapP", "EToB", "receiver_elements"}) {
            auto x = Indices(a_dir / (std::string(name) + ".bin"), a.at("index_bytes")), y = Indices(b_dir / (std::string(name) + ".bin"), b.at("index_bytes"));
            Require(x == y, "Upstream connectivity differs: " + std::string(name));
            if (std::string(name) == "receiver_elements") a["receiver_elements"] = x;
            else if (std::string(name) == "vmapM") local_map = x;
            else if (std::string(name) == "EToB") tags = x;
            else Write(source / (std::string(name) + ".bin"), x);
        }
        const auto positions = Read<float>(b_dir / "receiver_xyz.bin", size_t(int(a.at("receivers"))) * 3);
        a["receiver_xyz"] = Json::array();
        for (size_t i = 0; i < positions.size(); i += 3) a["receiver_xyz"].push_back({double(positions[i]), double(positions[i + 1]), double(positions[i + 2])});
        a["index_bytes"] = 4;
        Save(source / "mesh.json", a);
        int degree = a.at("degree"), n = a.at("nodes_per_element");
        std::ostringstream name;
        name << "tetN" << std::setw(2) << std::setfill('0') << degree << ".dat";
        std::ifstream in(checkout / "nodes" / name.str());
        Mat nodes(n, 3);
        std::string line;
        for (int axis = 0; axis < 3; ++axis) {
            bool found = false;
            while (std::getline(in, line))
                if (line == "Nodal " + std::string(1, "rst"[axis]) + "-coordinates") {
                    found = true;
                    break;
                }
            Require(found, "Missing upstream nodes");
            std::getline(in, line);
            Require(std::stoi(line) == n, "Wrong upstream node count");
            for (int i = 0; i < n; ++i) {
                std::getline(in, line);
                nodes(i, axis) = std::stod(line);
            }
        }
        {
            std::ofstream out(source / "nodes.dat");
            out << std::setprecision(17);
            for (int i = 0; i < n; ++i) out << nodes(i, 0) << ' ' << nodes(i, 1) << ' ' << nodes(i, 2) << '\n';
            Require(bool(out), "Cannot write nodes");
        }
        std::cout << "Preparing upstream " << case_name << std::endl;
        Reference const ref(source, source / "nodes.dat");
        Require(ref.Source.Vm == local_map, "Upstream local face ordering differs");
        Require(tags.size() == size_t(ref.Source.E) * 4, "Incorrect upstream boundary tag count");
        for (size_t f = 0; f < tags.size(); ++f) Require(tags[f] >= -1 && tags[f] <= 1 && (tags[f] > 0) == bool(ref.Source.Boundary[f * ref.Source.Nfp]), "Upstream boundary tags differ from rigid face maps");
        Save(dest / "reference-check.json", ref.Check());
        ref.Export(dest / "full/operator", 0);
        ref.Export(dest / "operator", 0, case_name == "cube_rigid" ? 9 : 12);
        Json hashes = Json::object();
        for (const auto &dir : {a_dir, b_dir})
            for (const auto &entry : fs::directory_iterator(dir))
                if (entry.is_regular_file()) hashes[entry.path().string()] = Hash(entry.path());
        Save(dest / "provenance.json", {{"capture_sha256", hashes}, {"nodes_sha256", Hash(checkout / "nodes" / name.str())}, {"source_sha256", Sources()}, {"runtime", Runtime()}});
    }
}
void Validate(const fs::path &output, const fs::path &inputs, const fs::path &executable) {
    VerifyFixtures();
    Require(fs::is_regular_file(executable), "Build DgTest before validation");
    NewDirectory(output);
    auto hashes = Sources();
    auto executable_hash = Hash(executable);
    Json commands = Json::array(), thermal = Json::array(), results;
    auto const native = [&](const fs::path &tables, const fs::path &dest, int steps = 1024, double dt = 0x1p-18, int repeats = 1) {
        int before = Thermal(output / "thermal-state.log");
        Require(before == 0, "Thermal pressure is elevated; retry after cooldown");
        std::ostringstream timestep;
        timestep << std::setprecision(17) << dt;
        std::vector<std::string> const argv{executable.string(), tables.string(), dest.string(), std::to_string(steps), timestep.str(), std::to_string(repeats)};
        commands.push_back(argv);
        Save(output / "commands.json", commands);
        std::cout << dest << std::endl;
        Require(Run(argv, dest.string() + ".log") == 0, "DgTest failed; see " + dest.string() + ".log");
        int after = Thermal(output / "thermal-state.log");
        thermal.push_back({{"output", dest.string()}, {"before", before}, {"after", after}});
        Save(output / "thermal.json", thermal);
        Require(after == 0, "Thermal pressure changed during a run");
    };
    auto const generated = output / "inputs";
    NewDirectory(generated);
    auto const primary = Operator(inputs, Fixtures[0]);
    CopyOperator(primary, generated / "replicated", 4);
    for (int i = 0; i < 3; ++i) {
        bool const physical = i < 2;
        const auto &f = Fixtures[physical ? i : 0];
        std::string const name = physical ? f.Name : "replicas";
        auto tables = physical ? Operator(inputs, f) : generated / "replicated", directory = output / name;
        NewDirectory(directory);
        native(tables, directory / "warm", 64);
        native(tables, directory / "native", physical ? 1024 : 512, 0x1p-18, 4);
        auto const timings = Load(directory / "native/result.json").at("runs");
        std::vector<double> gpu, wall;
        for (const auto &timing : timings) {
            gpu.push_back(timing.at("gpu_seconds"));
            wall.push_back(timing.at("wall_seconds_including_diagnostics"));
        }
        Json result = {{"timings", timings}, {"operator_sha256", Hash(tables / "operator.json")}, {"gpu_seconds", Median(gpu)}, {"wall_seconds", Median(wall)}};
        if (physical) {
            auto const record = directory / "native";
            result["analytic"] = Score(FixtureRoot / "meshes" / f.MeshName, record, Nodes(f));
            Accurate(result["analytic"]);
            result["versus_fp64"] = Compare(FixtureRoot / "cpu" / f.MeshName / "wadg/repeat1", record, tables);
            Gate(result["versus_fp64"], 2e-5);
        }
        results[name] = result;
    }
    const auto &f = Fixtures[0];
    auto const directory = output / "degree6";
    native(primary, directory / "half_dt", 2048, 0x1p-19, 2);
    auto half = Compare(directory / "native", directory / "half_dt", primary, 2);
    Gate(half, 2e-5);
    native(primary, directory / "long", 8192, 0x1p-18, 2);
    auto score = Score(FixtureRoot / "meshes" / f.MeshName, directory / "long", Nodes(f));
    Accurate(score);
    results["controls"] = {{"half_dt", half}, {"long_analytic", score}};
    CopyOperator(primary, generated / "tail", 1, true);
    native(generated / "tail", output / "tail", 16, 0x1p-18, 2);
    results["tail_passed"] = true;
    auto const original = Operator(inputs, f, false);
    native(original, output / "original_quadrature", 1024, 0x1p-18, 2);
    results["original_quadrature"] = Compare(FixtureRoot / "cpu" / f.MeshName / "wadg/repeat1", output / "original_quadrature", original);
    Gate(results["original_quadrature"], 2e-5);
    for (const std::string name : {"G", "L"}) {
        auto const bad = generated / ("bad_" + name);
        NewDirectory(bad);
        for (const auto &entry : fs::directory_iterator(primary)) fs::create_symlink(fs::absolute(entry.path()), bad / entry.path().filename());
        auto meta = Load(primary / "operator.json");
        auto data = Read<float>(primary / (name + ".bin"));
        int n = meta.at("nodes"), faces = meta.at("face_nodes");
        size_t const index = name == "G" ? 3 * n * n : Read<int32_t>(primary / "vmapM.bin").at(0) * faces + 1;
        data.at(index) += .01f;
        fs::remove(bad / (name + ".bin"));
        Write(bad / (name + ".bin"), data);
        std::vector<std::string> const argv{executable.string(), bad.string(), (output / ("bad_" + name)).string(), "1"};
        commands.push_back(argv);
        auto const log = output / ("bad_" + name + ".log");
        int const status = Run(argv, log);
        std::string const expected = name == "G" ? "Inconsistent weak volume transpose" : "Nonsymmetric weak face load";
        Require(status != 0 && Contents(log).find(expected) != std::string::npos, "Invalid operator was not rejected: " + name);
    }
    results["inconsistent_tables_rejected"] = true;
    CheckAnalytic(output / "analytic-checks");
    Save(output / "commands.json", commands);
    Require(Sources() == hashes && Hash(executable) == executable_hash, "Validation sources changed during run");
    Save(output / "acceptance.json", {{"passed", true}, {"results", results}});
    Save(output / "provenance.json", {{"source_sha256", hashes}, {"runtime", Runtime()}, {"executable_sha256", executable_hash}, {"fixture_manifest_sha256", Hash(FixtureRoot / "sha256.json")}, {"timing_note", "Nominal macOS thermal pressure before/after every successful native run; not a temperature or GPU clock measurement"}});
    Json artifacts = Json::object();
    for (const auto &entry : fs::recursive_directory_iterator(output))
        if (entry.is_regular_file()) artifacts[entry.path().lexically_relative(output).string()] = Hash(entry.path());
    Save(output / "artifact-sha256.json", artifacts);
}
} // namespace dg::reference
int main(int argc, char *const *argv) {
    using namespace dg::reference;
    try {
        setenv("VECLIB_MAXIMUM_THREADS", "1", 1);
        Eigen::setNbThreads(1);
        Require(argc >= 3, "Usage: DgReference prepare output | validate output [inputs] | reference case output [wadg|full_mass] | import-upstream captures checkout output | prepare-mesh mesh nodes output [offset=2] [order=0] | check output | score mesh record nodes | compare reference candidate tables [stride=1]");
        std::string const command = argv[1];
        if (command == "check") {
            Require(argc == 3, "Output required");
            CheckAnalytic(argv[2]);
        } else if (command == "prepare") {
            Require(argc == 3, "Output required");
            Prepare(argv[2]);
        } else if (command == "validate") {
            Require(argc == 3 || argc == 4, "Output and optional inputs required");
            Validate(argv[2], argc == 4 ? fs::path(argv[3]) : Project / "build/dg-inputs", Executable().parent_path() / "DgTest");
        } else if (command == "prepare-mesh") {
            Require(argc >= 5 && argc <= 7, "Mesh, nodes, output required");
            Reference const ref(argv[2], argv[3]);
            std::cout << ref.Check().dump(2) << std::endl;
            ref.Export(argv[4], argc > 5 ? std::stod(argv[5]) : 2, argc > 6 ? std::stoi(argv[6]) : 0);
        } else if (command == "score") {
            Require(argc == 5, "Mesh, record, nodes required");
            std::cout << Score(argv[2], argv[3], argv[4]).dump(2) << std::endl;
        } else if (command == "compare") {
            Require(argc == 5 || argc == 6, "Reference, candidate, tables required");
            std::cout << Compare(argv[2], argv[3], argv[4], argc > 5 ? std::stoi(argv[5]) : 1).dump(2) << std::endl;
        } else if (command == "reference") {
            Require(argc == 4 || argc == 5, "Case and output required");
            std::string const form = argc == 5 ? argv[4] : "wadg";
            Require(form == "wadg" || form == "full_mass", "Invalid reference form");
            Regenerate(argv[2], argv[3], form == "wadg");
        } else if (command == "import-upstream") {
            Require(argc == 5, "Captures, checkout, output required");
            ImportUpstream(argv[2], argv[3], argv[4]);
        } else throw std::runtime_error("Unknown command");
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
