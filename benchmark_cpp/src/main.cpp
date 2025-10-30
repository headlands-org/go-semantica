#include "benchmark_batch.h"
#include "benchmark_comprehensive.h"
#include "benchmark_isolated.h"
#include <iostream>
#include <string>
#include <cstring>
#include <thread>

void printUsage(const char* prog) {
    unsigned int numCPU = std::thread::hardware_concurrency();
    if (numCPU == 0) numCPU = 8; // fallback if detection fails

    std::cerr << "Usage: " << prog << " [options]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Benchmark embedding generation performance." << std::endl;
    std::cerr << std::endl;
    std::cerr << "Required flags:" << std::endl;
    std::cerr << "  -model string" << std::endl;
    std::cerr << "        Path to GGUF model file" << std::endl;
    std::cerr << "  -duration int" << std::endl;
    std::cerr << "        Benchmark duration in seconds (not required for comprehensive mode)" << std::endl;
    std::cerr << "  -mode string" << std::endl;
    std::cerr << "        Benchmark mode: 'batch', 'isolated', or 'comprehensive'" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Optional flags:" << std::endl;
    std::cerr << "  -batch-size int" << std::endl;
    std::cerr << "        Batch size for batch mode (default " << (numCPU * 4) << ")" << std::endl;
    std::cerr << "  -workers int" << std::endl;
    std::cerr << "        Number of worker goroutines (default " << (numCPU * 2) << ")" << std::endl;
    std::cerr << "  -threads int" << std::endl;
    std::cerr << "        llama.cpp CPU threads per context (default " << (numCPU * 2) << ")" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Example commands:" << std::endl;
    std::cerr << "  # Batch mode with 10 second run" << std::endl;
    std::cerr << "  " << prog << " -model model/embeddinggemma-300m-Q8_0.gguf -mode batch -duration 10" << std::endl;
    std::cerr << std::endl;
    std::cerr << "  # Isolated mode with custom workers and duration" << std::endl;
    std::cerr << "  " << prog << " -model model/embeddinggemma-300m-Q8_0.gguf -mode isolated -duration 30 -workers 8" << std::endl;
    std::cerr << std::endl;
    std::cerr << "  # Comprehensive benchmark for README documentation" << std::endl;
    std::cerr << "  " << prog << " -model model/embeddinggemma-300m-Q8_0.gguf -mode comprehensive" << std::endl;
}

// Parse command-line flags
struct Config {
    std::string modelPath;
    std::string mode;
    int duration = 0;
    int batchSize = -1;  // -1 means use default (NumCPU * 4)
    int workers = -1;    // -1 means use default (NumCPU * 2)
    int threads = -1;    // -1 means use default (NumCPU * 2)
    bool valid = false;
};

Config parseArgs(int argc, char** argv) {
    Config config;

    // Get number of CPUs for defaults
    unsigned int numCPU = std::thread::hardware_concurrency();
    if (numCPU == 0) numCPU = 8; // fallback if detection fails

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        // Check for help flags
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        }

        // Support both -flag=value and -flag value formats
        std::string flag, value;
        size_t eqPos = arg.find('=');

        if (eqPos != std::string::npos) {
            // -flag=value format
            flag = arg.substr(0, eqPos);
            value = arg.substr(eqPos + 1);
        } else {
            // -flag value format (get value from next arg)
            flag = arg;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                value = argv[i + 1];
                ++i; // skip next arg since we used it as value
            }
        }

        if (flag == "-model") {
            config.modelPath = value;
        } else if (flag == "-mode") {
            config.mode = value;
        } else if (flag == "-duration") {
            if (value.empty()) {
                std::cerr << "Error: -duration requires a value" << std::endl;
                return config;
            }
            config.duration = std::stoi(value);
        } else if (flag == "-batch-size") {
            if (value.empty()) {
                std::cerr << "Error: -batch-size requires a value" << std::endl;
                return config;
            }
            config.batchSize = std::stoi(value);
        } else if (flag == "-workers") {
            if (value.empty()) {
                std::cerr << "Error: -workers requires a value" << std::endl;
                return config;
            }
            config.workers = std::stoi(value);
        } else if (flag == "-threads") {
            if (value.empty()) {
                std::cerr << "Error: -threads requires a value" << std::endl;
                return config;
            }
            config.threads = std::stoi(value);
        } else {
            std::cerr << "Error: Unknown flag: " << flag << std::endl;
            return config;
        }
    }

    // Apply defaults if not set
    if (config.batchSize == -1) {
        config.batchSize = numCPU * 4;
    }
    if (config.workers == -1) {
        config.workers = numCPU * 2;
    }
    if (config.threads == -1) {
        config.threads = static_cast<int>(numCPU) * 2;
        if (config.threads <= 0) {
            config.threads = 1;
        }
    }

    // Validate required flags
    if (config.modelPath.empty()) {
        std::cerr << "Error: -model is required" << std::endl;
        return config;
    }

    if (config.mode.empty()) {
        std::cerr << "Error: -mode is required" << std::endl;
        return config;
    }

    // Validate mode
    if (config.mode != "batch" && config.mode != "isolated" && config.mode != "comprehensive") {
        std::cerr << "Error: -mode must be 'batch', 'isolated', or 'comprehensive'" << std::endl;
        return config;
    }

    // Duration is required for batch and isolated modes
    if ((config.mode == "batch" || config.mode == "isolated") && config.duration <= 0) {
        std::cerr << "Error: -duration must be greater than 0 for batch/isolated modes" << std::endl;
        return config;
    }

    config.valid = true;
    return config;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    Config config = parseArgs(argc, argv);
    if (!config.valid) {
        std::cerr << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    try {
        if (config.mode == "batch") {
            return benchmark::runBatchMode(config.modelPath, config.duration, config.batchSize, config.threads);
        } else if (config.mode == "isolated") {
            auto results = benchmark::runIsolatedMode(config.modelPath, config.duration, config.workers, config.threads);
            // runIsolatedMode prints its own output, just check if we got results
            return results.empty() ? 1 : 0;
        } else if (config.mode == "comprehensive") {
            benchmark::runComprehensiveMode(config.modelPath, config.threads);
            return 0;
        } else {
            std::cerr << "Unknown mode: " << config.mode << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
