
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <boost/program_options.hpp>
#include "gomoku/selfplay.h"


namespace po = boost::program_options;
namespace fs = std::filesystem;




int main(int argc, char *argv[]) {
    
    gomoku::selfplay::Server::Config config;
    po::options_description gen_cfg("generic config");
    gen_cfg.add_options()
        ("help,h", "usage")
    ;
    po::options_description mcts_cfg = 
        mcts::GetMCTSConfig(config.mcts_cfg);
    po::options_description selfpaly_cfg 
        = gomoku::selfplay::GetSelfplayConfig(config);
    po::options_description options;
    options.add(mcts_cfg).add(selfpaly_cfg);

    
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(argc, argv)
        .options(gen_cfg)
        .allow_unregistered()
        .run();
    try {
        po::store(parsed, vm);
        po::notify(vm);
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
    try {
        parsed = po::command_line_parser(unrec).options(options).run();
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

    gomoku::selfplay::Server server(config);
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
