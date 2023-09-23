
#include <algorithm>
#include <torch/torch.h>
#include "gomoku/evaluator.h"


namespace gomoku {

GomokuEvaluator::GomokuEvaluator(torch::jit::script::Module&& model_)
:model(model_), device(torch::kCPU) {
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
    }
    model.eval();
    model.to(device);
    if (torch::cuda::is_available()) {
        torch::NoGradGuard();
        std::vector<torch::jit::IValue> warmup({
            torch::randn({2, 3, SIZE, SIZE}, torch::kFloat32).to(device),
            torch::ones({2}, torch::kInt64).to(device)
        });
        model(warmup);
        model(warmup);
    }
}


std::vector<Output> GomokuEvaluator::EvaluateBatch(std::vector<Input>& inputs) {
    int size = inputs.size();
    std::vector<torch::Tensor> states, turns, masks;
    torch::NoGradGuard();

    for (Input& input: inputs) {
        states.push_back(std::move(input.state));
        turns.push_back(std::move(input.turn));
        masks.push_back(std::move(input.mask));
    }
    
    torch::Tensor input_tensor = torch::stack(std::move(states)).to(device);
    torch::Tensor turn_tensor = torch::concatenate(std::move(turns)).to(device);
    torch::Tensor mask_tensor = torch::stack(std::move(masks)).to(device);

    std::vector<torch::jit::IValue> batch;
    batch.push_back(std::move(input_tensor));
    batch.push_back(std::move(turn_tensor));

    torch::jit::IValue out = model.forward(batch);
    torch::Tensor probs = out.toTuple()->elements()[0].toTensor();
    torch::Tensor results = out.toTuple()->elements()[1].toTensor();

    probs = torch::masked_fill(probs, mask_tensor, -1e+9);
    probs = torch::nn::functional::softmax(probs, 1);
    probs = probs.to(torch::kCPU);
    results = results.to(torch::kCPU);

    float* results_ptr = results.data_ptr<float>();
    std::vector<Output> ret;
    for (int i = 0; i < size; i++) {
        ret.emplace_back(results_ptr[i], probs.index({i}));
    }
    return ret;
}


Input GomokuEvaluator::Preprocess(const Board& board) const {
    Input ret;
    ret.state = torch::from_blob(
        (void*)board.GetDataPtr(),
        {3, SIZE, SIZE},
        torch::TensorOptions().dtype(torch::kInt8)
    ).to(torch::kFloat32);
    ret.turn = ((board.GetTurn() == BLACK) ? 
        torch::zeros(1, torch::TensorOptions().dtype(torch::kInt64)) :
        torch::ones(1, torch::TensorOptions().dtype(torch::kInt64)));
    ret.mask = torch::from_blob(
        (void*)board.GetDataPtr(),
        {SIZE, SIZE},
        torch::TensorOptions().dtype(torch::kInt8)
    ).to(torch::kBool).reshape({SIZE * SIZE});
    ret.mask = torch::logical_not(ret.mask);
    return ret;
}

Evaluation GomokuEvaluator::Postprocess
(Output& output, const Board& board) const {
    Reward r = std::clamp<double>(output.first, -1, 1);
    if (board.GetTurn() == WHITE)
        r = -r;
    
    std::vector<std::pair<Action, Prob>> probs;

    if (board.GetTurnElapsed() == 0) {
        probs.emplace_back(Coord2Action(SIZE / 2, SIZE / 2), 1.);
    }
    else {
        const int8_t* empty_plane = board.GetDataPtr();
        float* pred_ptr = output.second.data_ptr<float>();

        for (int i = 0; i < SIZE*SIZE; i++) {
            if (empty_plane[i] == 1) {
                probs.emplace_back((Action)i, (Prob)pred_ptr[i]);
            }
        }
    }

    return Evaluation(r, std::move(probs));
}


}
