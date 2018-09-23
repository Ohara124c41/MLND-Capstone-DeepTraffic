
//<![CDATA[

// a few things don't have var in front of them - they update already existing variables the game needs
lanesSide = 0;
patchesAhead = 1;
patchesBehind = 0;
trainIterations = 10000;

// the number of other autonomous vehicles controlled by your network
otherAgents = 0; // max of 10

var num_inputs = (lanesSide * 2 + 1) * (patchesAhead + patchesBehind);
var num_actions = 5;
var temporal_window = 3;
var network_size = num_inputs * temporal_window + num_actions * temporal_window + num_inputs;

var layer_defs = [];
    layer_defs.push({
    type: 'input',
    out_sx: 1,
    out_sy: 1,
    out_depth: network_size
});
layer_defs.push({
    type: 'fc',
    num_neurons: 1,
    activation: 'relu'
});
layer_defs.push({
    type: 'regression',
    num_neurons: num_actions
});

var tdtrainer_options = {
    learning_rate: 0.001,
    momentum: 0.0,
    batch_size: 64,
    l2_decay: 0.01
};

var opt = {};
opt.temporal_window = temporal_window;
opt.experience_size = 3000;
opt.start_learn_threshold = 500;
opt.gamma = 0.7;
opt.learning_steps_total = 10000;
opt.learning_steps_burnin = 1000;
opt.epsilon_min = 0.0;
opt.epsilon_test_time = 0.0;
opt.layer_defs = layer_defs;
opt.tdtrainer_options = tdtrainer_options;

brain = new deepqlearn.Brain(num_inputs, num_actions, opt);

learn = function (state, lastReward) {
brain.backward(lastReward);
var action = brain.forward(state);

draw_net();
draw_stats();

return action;
}

//]]>

/*###########*/
if (brain) {
brain.value_net.fromJSON({"layers":[{"out_depth":19,"out_sx":1,"out_sy":1,"layer_type":"input"},{"out_depth":1,"out_sx":1,"out_sy":1,"layer_type":"fc","num_inputs":19,"l1_decay_mul":0,"l2_decay_mul":1,"filters":[{"sx":1,"sy":1,"depth":19,"w":{"0":0.10392464179889462,"1":0.48544930062379626,"2":0.03217968599918736,"3":0.22712493313809548,"4":-0.3820887512367659,"5":0.2406270727285015,"6":0.09273283371656794,"7":0.14218866695042456,"8":0.290479241922673,"9":0.3736890726887165,"10":0.1139025822570971,"11":0.1638716961380827,"12":-0.31467439272838116,"13":0.49174111434239204,"14":0.5237890097415374,"15":-0.030382141023136463,"16":-0.03536122944123721,"17":-0.24765808731301744,"18":0.07833547513272797}}],"biases":{"sx":1,"sy":1,"depth":1,"w":{"0":0.1}}},{"out_depth":1,"out_sx":1,"out_sy":1,"layer_type":"relu"},{"out_depth":5,"out_sx":1,"out_sy":1,"layer_type":"fc","num_inputs":1,"l1_decay_mul":0,"l2_decay_mul":1,"filters":[{"sx":1,"sy":1,"depth":1,"w":{"0":-0.6455855339710701}},{"sx":1,"sy":1,"depth":1,"w":{"0":-0.20053912344652838}},{"sx":1,"sy":1,"depth":1,"w":{"0":-0.3692329694159794}},{"sx":1,"sy":1,"depth":1,"w":{"0":0.5283168116281813}},{"sx":1,"sy":1,"depth":1,"w":{"0":-0.6314732999528674}}],"biases":{"sx":1,"sy":1,"depth":5,"w":{"0":0,"1":0,"2":0,"3":0,"4":0}}},{"out_depth":5,"out_sx":1,"out_sy":1,"layer_type":"regression","num_inputs":5}]});
}