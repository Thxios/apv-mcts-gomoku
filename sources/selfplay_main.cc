
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <boost/program_options.hpp>
#include "gomoku/server.h"


namespace po = boost::program_options;
namespace fs = std::filesystem;

using gomoku::selfplay::Server;


po::options_description GetMCTSConfig(Server::Config& cfg) {
    po::options_description desc("MCTS config");
    desc.add_options()
        (
            "model_path", 
            po::value<fs::path>(&cfg.model_path)->required(), 
            "torch jit module path"
        )
        (
            "n_threads", 
            po::value<size_t>(&cfg.mcts_cfg.n_threads)->default_value(4),
            "number of search threads per tree"
        )
        (
            "virtual_loss", 
            po::value<int>(&cfg.mcts_cfg.virtual_loss)->default_value(3),
            "virtual loss for MCTS search"
        )
        (
            "p_uct", 
            po::value<double>(&cfg.mcts_cfg.p_uct)->default_value(5.),
            "p_uct for MCTS search"
        )
    ;
    return desc;
}


po::options_description GetSelfplayConfig(Server::Config& cfg) {
    po::options_description desc("Selfplay config");
    desc.add_options()
        (
            "out_dir", 
            po::value<fs::path>(&cfg.out_dir)->required(), 
            "output directory"
        )
        (
            "starting_index", 
            po::value<size_t>(&cfg.starting_index)->default_value(0),
            "starting index for logging"
        )
        (
            "max_games", 
            po::value<size_t>(&cfg.max_games)->default_value(100),
            "maximum number of games to be played"
        )
        (
            "n_workers", 
            po::value<size_t>(&cfg.n_workers)->default_value(1),
            "number of games played simultaneously"
        )

        (
            "n_searches", 
            po::value<size_t>(&cfg.sp_cfg.compute_budget)->default_value(800),
            "number of MCTS searches per each move"
        )
        (
            "sample_steps", 
            po::value<size_t>(&cfg.sp_cfg.sample_steps)->default_value(15),
            "number of moves to be weighted-random sampled"
        )
        (
            "noise_steps", 
            po::value<size_t>(&cfg.sp_cfg.noise_steps)->default_value(3),
            "number of moves to be applied dirichlet noise"
        )
        (
            "noise_eps", 
            po::value<double>(&cfg.sp_cfg.noise_eps)->default_value(0.25),
            "epsilon for selfplay noise"
        )
        (
            "noise_alpha", 
            po::value<double>(&cfg.sp_cfg.noise_alpha)->default_value(0.03),
            "dirichlet alpha for selfplay noise"
        )
    ;
    return desc;
}



int main(int argc, char *argv[]) {
    
    Server::Config config;
    po::options_description gen_cfg("generic config");
    gen_cfg.add_options()
        ("help,h", "usage")
    ;
    po::options_description mcts_cfg = GetMCTSConfig(config);
    po::options_description selfpaly_cfg = GetSelfplayConfig(config);
    po::options_description options;
    options.add(mcts_cfg).add(selfpaly_cfg);

    
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(argc, argv)
        .options(gen_cfg)
        .allow_unregistered()
        .run();
    try {
        po::store(parsed, vm);
    }
    catch (po::error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    if (vm.count("help")) {
        std::cout << gen_cfg << std::endl;
        std::cout << mcts_cfg << std::endl;
        std::cout << selfpaly_cfg << std::endl;
        return 0;
    }

    
    std::vector<std::string> unrec
        = po::collect_unrecognized(parsed.options, po::include_positional);
    parsed = po::command_line_parser(unrec)
        .options(options)
        .run();
    try {
        po::store(parsed, vm);
        po::notify(vm);
    }
    catch (po::error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    std::cout << "===== Selfplay Settings =====\n";
    std::cout << config << "\n";
    std::cout << "=============================" << std::endl;

    std::cout << std::endl;

    Server server(config);
    try {
        server.LoadEvauator();
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model" << std::endl;
        std::cerr << e.what() << std::endl;
        return 1;
    }
    
    std::cout << std::endl;
    
    server.Run();

}
