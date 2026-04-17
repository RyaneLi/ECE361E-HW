#!/bin/bash

# This script is used to generate commands to run cloud.py and device.py on edge devices
# Change the parameters below and run "bash run.bash" on terminal
# It will also run "generate_configs.py" based on the given parameters
# "dense1_anchor",
#     "conv2d1_anchor",
#     "conv2d1_leakyrelu",
#     "conv2d1_maxpool",
#     "conv2d1_batchnorm",
#     "conv2d1_groupnorm",
#     "conv2d1_residual_lite",
#     "depthwise1_anchor",
#     "depthwise1_leakyrelu",
#     "depthwise1_maxpool",
#     "depthwise1_batchnorm",
#     "depthwise1_groupnorm",
#     "depthwise1_residual_lite",


############### TODO CHANGE
cloud_ip="10.157.72.74" # Change every time when new VPN is connected

# Defaults used by all experiments unless explicitly overridden in experiment_configs.
default_model_name="conv5small"
default_learning_rate=0.01
loss_func_name="cross_entropy"
default_loss_type="fedavg"   # fedavg | fedprox | fedmax
default_mu=1.0                # fedprox coefficient
default_beta=10.0             # fedmax coefficient

declare -a experiment_configs=(
# experiment | run | data_iid | model_name | learning_rate | loss_type | mu | beta | rpi_local_epochs | mc1_local_epochs
  "100 1 false simplefc 0.01 fedavg 1.0 10.0 1 1"
  "101 1 false simplefc 0.01 fedavg 1.0 10.0 2 1"
  "102 1 false simplecnn_singleconv 0.01 fedavg 1.0 10.0 1 1"
  "103 1 false simplecnn_singleconv 0.01 fedavg 1.0 10.0 2 1"
  "104 1 false simplecnn_singleconv 0.01 fedavg 1.0 10.0 3 1"
  "105 1 false simplecnn_singleconv_1_16 0.01 fedavg 1.0 10.0 1 1"
  "106 1 false simplecnn_singleconv_1_8 0.01 fedavg 1.0 10.0 1 1"
  "107 1 false simplecnn_singleconv_1_8 0.01 fedavg 1.0 10.0 1 2"
  "108 1 false simplecnn_singleconv_1_4 0.01 fedavg 1.0 10.0 1 1"
  "109 1 false simplecnn_singleconv_1_4 0.01 fedavg 1.0 10.0 1 2"
  "110 1 false simplecnn_small 0.01 fedavg 1.0 10.0 1 1"
  "111 1 false simplecnn_small 0.01 fedavg 1.0 10.0 2 1"
  "112 1 false simplecnn_small 0.01 fedavg 1.0 10.0 3 1"
  "113 1 false simplecnn_small_12_24 0.01 fedavg 1.0 10.0 1 1"
  "114 1 false simplecnn_small_12_24 0.01 fedavg 1.0 10.0 2 1"
  "115 1 false simplecnn_small_8_16 0.01 fedavg 1.0 10.0 1 1"
  "116 1 false simplecnn_small_8_16 0.01 fedavg 1.0 10.0 2 1"
  "117 1 false simplecnn_small_4_8 0.01 fedavg 1.0 10.0 1 1"
  "118 1 false simplecnn_small_4_8 0.01 fedavg 1.0 10.0 1 2"
  "119 1 false simplecnn 0.01 fedavg 1.0 10.0 1 1"
  "120 1 false simplecnn 0.01 fedavg 1.0 10.0 2 1"
  "121 1 false dense1_anchor 0.01 fedavg 1.0 10.0 1 1"
  "122 1 false dense1_anchor 0.01 fedavg 1.0 10.0 1 2"
  "123 1 false conv2d1_anchor 0.01 fedavg 1.0 10.0 1 1"
  "124 1 false conv2d1_anchor 0.01 fedavg 1.0 10.0 1 2"
  "125 1 false conv2d1_leakyrelu 0.01 fedavg 1.0 10.0 1 1"
  "126 1 false conv2d1_leakyrelu 0.01 fedavg 1.0 10.0 1 2"
  "127 1 false conv2d1_maxpool 0.01 fedavg 1.0 10.0 1 1"
  "128 1 false conv2d1_batchnorm 0.01 fedavg 1.0 10.0 1 1"
  "129 1 false conv2d1_groupnorm 0.01 fedavg 1.0 10.0 1 1"
  "130 1 false conv2d1_groupnorm 0.01 fedavg 1.0 10.0 1 2"
  "131 1 false conv2d1_residual_lite 0.01 fedavg 1.0 10.0 1 1"
  "132 1 false conv2d1_residual_lite 0.01 fedavg 1.0 10.0 1 2"
  "133 1 false depthwise1_anchor 0.01 fedavg 1.0 10.0 1 1"
  "134 1 false depthwise1_leakyrelu 0.01 fedavg 1.0 10.0 1 1"
  "135 1 false depthwise1_maxpool 0.01 fedavg 1.0 10.0 1 1"
  "136 1 false depthwise1_batchnorm 0.01 fedavg 1.0 10.0 1 1"
  "137 1 false depthwise1_groupnorm 0.01 fedavg 1.0 10.0 1 1"
  "138 1 false depthwise1_residual_lite 0.01 fedavg 1.0 10.0 1 1"
  "139 1 false simplecnn_singleconv 0.01 fedprox 1.0 10.0 1 1"
  "140 1 false simplecnn_singleconv 0.01 fedmax 1.0 10.0 1 1"
  "141 1 false simplecnn_small 0.01 fedprox 1.0 10.0 1 1"
  "142 1 false simplecnn_small 0.01 fedmax 1.0 10.0 1 1"
  "143 1 false simplecnn_small_12_24 0.01 fedprox 1.0 10.0 1 1"
  "144 1 false simplecnn_small_12_24 0.01 fedmax 1.0 10.0 1 1"
  "145 1 false simplecnn_small_8_16 0.01 fedprox 1.0 10.0 1 1"
  "146 1 false simplecnn_small_8_16 0.01 fedmax 1.0 10.0 1 1"
  "147 1 false simplecnn_small_4_8 0.01 fedprox 1.0 10.0 1 1"
  "148 1 false simplecnn_small_4_8 0.01 fedmax 1.0 10.0 1 1"
  "149 1 false dense1_anchor_hidden_256 0.01 fedavg 1.0 10.0 1 1"
  "150 1 false dense1_anchor_deep 0.01 fedavg 1.0 10.0 1 1"
  "151 1 false dense1_anchor_conv_lite 0.01 fedavg 1.0 10.0 1 1"
  "152 1 false dense1_anchor_conv_lite_v2 0.01 fedavg 1.0 10.0 1 1"
  "153 1 false simplecnn_20_40 0.01 fedavg 1.0 10.0 1 1"
  "154 1 false simplecnn_20_40_batchnorm 0.01 fedavg 1.0 10.0 1 1"
  "155 1 false simplecnn_24_48 0.01 fedavg 1.0 10.0 1 1"
  "156 1 false simplecnn_bottleneck 0.01 fedavg 1.0 10.0 1 1"
  "157 1 false simplecnn_small_4_16 0.01 fedavg 1.0 10.0 1 1"
  "158 1 false simplecnn_small_4_8_deep 0.01 fedavg 1.0 10.0 1 1"
  "159 1 false simplecnn_small_4_8_deep_groupnorm 0.01 fedavg 1.0 10.0 1 1"
  "160 1 false simplecnn_small_8_16_deep 0.01 fedavg 1.0 10.0 1 1"
  "161 1 false simplecnn_small_12_24_deep 0.01 fedavg 1.0 10.0 1 1"
  "162 1 false simplecnn_singleconv_1_4_twoconv 0.01 fedavg 1.0 10.0 1 1"
  "163 1 false simplecnn_small_4_8_deep 0.01 fedmax 1.0 15.0 1 1"
  "164 1 false refined_simplecnn_small_4_8_deep 0.01 fedavg 1.0 10.0 1 1"
  "165 1 false refined_simplecnn_small_4_8_deep 0.01 fedmax 1.0 10.0 1 1"
  "166 1 false refined_simplecnn_small_4_8_deep 0.01 fedprox 1.0 10.0 1 1"
  "167 1 false simplecnn_small_4_8_deep_groupnorm 0.01 fedmax 1.0 15.0 1 1"
  "168 1 false refined_simplecnn_small_4_8_deeper 0.01 fedavg 1.0 10.0 1 1"
  "169 1 false refined_simplecnn_small_4_8_deeper 0.01 fedmax 1.0 10.0 1 1"
  "170 1 false refined_simplecnn_small_4_8_deeper 0.01 fedprox 1.0 10.0 1 1"
  "171 1 false simplecnn_small_4_16 0.01 fedmax 1.0 15.0 1 1"
  "172 1 false refined_simplecnn_small_4_16 0.01 fedavg 1.0 10.0 1 1"
  "173 1 false refined_simplecnn_small_4_16 0.01 fedmax 1.0 10.0 1 1"
  "174 1 false refined_simplecnn_small_4_16 0.01 fedprox 1.0 10.0 1 1"
  "175 1 false simplecnn_20_40_batchnorm 0.02 fedavg 1.0 10.0 1 1"
  "176 1 false simplecnn_20_40_batchnorm 0.01 fedmax 1.0 15.0 1 1"
  "177 1 false simplecnn_20_40_batchnorm 0.02 fedmax 1.0 15.0 1 1"
  "178 1 false refined_simplecnn_20_40_batchnorm 0.01 fedavg 1.0 10.0 1 1"
  "179 1 false refined_simplecnn_20_40_batchnorm 0.01 fedmax 1.0 15.0 1 1"
  "180 1 false refined_simplecnn_20_40_batchnorm 0.01 fedprox 1.0 10.0 1 1"
  "181 1 false simplecnn_small_4_8 0.01 fedprox 1.5 10.0 1 1"
  "182 1 false simplecnn_small_4_8 0.01 fedmax 1.0 15.0 1 1"
  "183 1 false refined_simplecnn_small_4_8 0.01 fedavg 1.0 10.0 1 1"
  "184 1 false refined_simplecnn_small_4_8 0.01 fedmax 1.0 10.0 1 1"
  "185 1 false refined_simplecnn_small_4_8 0.01 fedprox 1.0 10.0 1 1"
  "186 1 false simplecnn_small_8_16 0.01 fedprox 1.5 10.0 1 1"
  "187 1 false refined_simplecnn_small_8_16 0.01 fedavg 1.0 10.0 1 1"
  "188 1 false refined_simplecnn_small_8_16 0.01 fedprox 1.0 10.0 1 1"
  "189 1 false refined_simplecnn_small_8_16 0.01 fedmax 1.0 10.0 1 1"
  "190 1 false simplecnn_24_48 0.01 fedmax 1.0 15.0 1 1"
  "191 1 false refined_simplecnn_24_48 0.01 fedavg 1.0 10.0 1 1"
  "192 1 false refined_simplecnn_24_48 0.01 fedmax 1.0 10.0 1 1"
  "193 1 false refined_simplecnn_24_48 0.01 fedprox 1.0 10.0 1 1"
  "194 1 false simplecnn_20_40 0.01 fedmax 1.0 15.0 1 1"
  "195 1 false refined_simplecnn_20_40 0.01 fedavg 1.0 10.0 1 1"
  "196 1 false refined_simplecnn_20_40 0.01 fedmax 1.0 10.0 1 1"
 
# CHAMPION VARIANTS
# Fine tuning first
  "197 1 false champion_sword 0.009 fedprox 1.0 10.0 1 1"
  "198 1 false champion_sword 0.010 fedprox 0.9 10.0 1 1"
  "199 1 false champion_sword 0.010 fedprox 1.0 10.0 1 1"
  "200 1 false champion_sword 0.010 fedprox 1.1 10.0 1 1"
  "201 1 false champion_sword 0.011 fedprox 0.9 10.0 1 1"
  "202 1 false champion_sword 0.011 fedprox 1.0 10.0 1 1"
  "203 1 false champion_sword 0.011 fedprox 1.1 10.0 1 1"
  "204 1 false champion_sword 0.012 fedprox 1.1 10.0 1 1"

# Now exploration
  "205 1 false champion_explorer 0.01 fedavg 1.0 10.0 1 1"
  "206 1 false champion_explorer 0.01 fedprox 1.0 10.0 1 1"
  "207 1 false champion_explorer 0.01 fedmax 1.0 10.0 1 1"

  "208 1 false champion_explorer_v2 0.01 fedavg 1.0 10.0 1 1"
  "209 1 false champion_explorer_v2 0.01 fedprox 1.0 10.0 1 1"
  "210 1 false champion_explorer_v2 0.01 fedmax 1.0 10.0 1 1"

  "211 1 false champion_fighter 0.01 fedavg 1.0 10.0 1 1"
  "212 1 false champion_fighter 0.01 fedprox 1.0 10.0 1 1"
  "213 1 false champion_fighter 0.01 fedmax 1.0 10.0 1 1"

  "214 1 false champion_fighter_v2 0.01 fedavg 1.0 10.0 1 1"
  "215 1 false champion_fighter_v2 0.01 fedprox 1.0 10.0 1 1"
  "216 1 false champion_fighter_v2 0.01 fedmax 1.0 10.0 1 1"

  "217 1 false champion_rogue 0.01 fedavg 1.0 10.0 1 1"
  "218 1 false champion_rogue 0.01 fedprox 1.0 10.0 1 1"
  "219 1 false champion_rogue 0.01 fedmax 1.0 10.0 1 1"

  "220 1 false champion_rogue_v2 0.01 fedavg 1.0 10.0 1 1"
  "221 1 false champion_rogue_v2 0.01 fedprox 1.0 10.0 1 1"
  "222 1 false champion_rogue_v2 0.01 fedmax 1.0 10.0 1 1"

  "223 1 false champion_knife 0.01 fedavg 1.0 10.0 1 1"
  "224 1 false champion_knife 0.01 fedprox 1.0 10.0 1 1"
  "225 1 false champion_knife 0.01 fedmax 1.0 10.0 1 1"

  "226 1 false champion_knife_v2 0.01 fedavg 1.0 10.0 1 1"
  "227 1 false champion_knife_v2 0.01 fedprox 1.0 10.0 1 1"
  "228 1 false champion_knife_v2 0.01 fedmax 1.0 10.0 1 1"

  "229 1 false champion_dagger 0.01 fedavg 1.0 10.0 1 1"
  "230 1 false champion_dagger 0.01 fedprox 1.0 10.0 1 1"
  "231 1 false champion_dagger 0.01 fedmax 1.0 10.0 1 1"

  "232 1 false champion_dagger_v2 0.01 fedavg 1.0 10.0 1 1"
  "233 1 false champion_dagger_v2 0.01 fedprox 1.0 10.0 1 1"
  "234 1 false champion_dagger_v2 0.01 fedmax 1.0 10.0 1 1"

  "235 1 false champion_blade 0.01 fedavg 1.0 10.0 1 1"
  "236 1 false champion_blade 0.01 fedprox 1.0 10.0 1 1"
  "237 1 false champion_blade 0.01 fedmax 1.0 10.0 1 1"

  "238 1 false champion_blade_v2 0.01 fedavg 1.0 10.0 1 1"
  "239 1 false champion_blade_v2 0.01 fedprox 1.0 10.0 1 1"
  "240 1 false champion_blade_v2 0.01 fedmax 1.0 10.0 1 1"
# Fixed some bad champion variants and now vanguard variants
  "241 1 false champion_fighter 0.01 fedavg 1.0 10.0 1 1"
  "242 1 false champion_fighter 0.01 fedprox 1.0 10.0 1 1"
  "243 1 false champion_fighter 0.01 fedmax 1.0 10.0 1 1"

  "244 1 false champion_fighter_v2 0.01 fedavg 1.0 10.0 1 1"
  "245 1 false champion_fighter_v2 0.01 fedprox 1.0 10.0 1 1"
  "246 1 false champion_fighter_v2 0.01 fedmax 1.0 10.0 1 1"

  "247 1 false champion_rogue 0.01 fedavg 1.0 10.0 1 1"
  "248 1 false champion_rogue 0.01 fedprox 1.0 10.0 1 1"
  "249 1 false champion_rogue 0.01 fedmax 1.0 10.0 1 1"

  "250 1 false champion_rogue_v2 0.01 fedavg 1.0 10.0 1 1"
  "251 1 false champion_rogue_v2 0.01 fedprox 1.0 10.0 1 1"
  "252 1 false champion_rogue_v2 0.01 fedmax 1.0 10.0 1 1"

  "253 1 false champion_knife 0.01 fedavg 1.0 10.0 1 1"
  "254 1 false champion_knife 0.01 fedprox 1.0 10.0 1 1"
  "255 1 false champion_knife 0.01 fedmax 1.0 10.0 1 1"

  "256 1 false champion_knife_v2 0.01 fedavg 1.0 10.0 1 1"
  "257 1 false champion_knife_v2 0.01 fedprox 1.0 10.0 1 1"
  "258 1 false champion_knife_v2 0.01 fedmax 1.0 10.0 1 1"

  "259 1 false champion_dagger 0.01 fedavg 1.0 10.0 1 1"
  "260 1 false champion_dagger 0.01 fedprox 1.0 10.0 1 1"
  "261 1 false champion_dagger 0.01 fedmax 1.0 10.0 1 1"

  "262 1 false champion_dagger_v2 0.01 fedavg 1.0 10.0 1 1"
  "263 1 false champion_dagger_v2 0.01 fedprox 1.0 10.0 1 1"
  "264 1 false champion_dagger_v2 0.01 fedmax 1.0 10.0 1 1"

# VANGUARD VARIANTS
  "265 1 false vanguard_sword_turbo 0.01 fedavg 1.0 10.0 1 1"
  "266 1 false vanguard_sword_turbo 0.01 fedprox 1.0 10.0 1 1"
  "267 1 false vanguard_sword_turbo 0.01 fedmax 1.0 10.0 1 1"

  "268 1 false vanguard_sword_midboost 0.01 fedavg 1.0 10.0 1 1"
  "269 1 false vanguard_sword_midboost 0.01 fedprox 1.0 10.0 1 1"
  "270 1 false vanguard_sword_midboost 0.01 fedmax 1.0 10.0 1 1"

  "271 1 false vanguard_sword_eco 0.01 fedavg 1.0 10.0 1 1"
  "272 1 false vanguard_sword_eco 0.01 fedprox 1.0 10.0 1 1"
  "273 1 false vanguard_sword_eco 0.01 fedmax 1.0 10.0 1 1"

  "274 1 false vanguard_explorerv2_turbo 0.01 fedavg 1.0 10.0 1 1"
  "275 1 false vanguard_explorerv2_turbo 0.01 fedprox 1.0 10.0 1 1"
  "276 1 false vanguard_explorerv2_turbo 0.01 fedmax 1.0 10.0 1 1"

  "277 1 false vanguard_explorerv2_depthwise_eco 0.01 fedavg 1.0 10.0 1 1"
  "278 1 false vanguard_explorerv2_depthwise_eco 0.01 fedprox 1.0 10.0 1 1"
  "279 1 false vanguard_explorerv2_depthwise_eco 0.01 fedmax 1.0 10.0 1 1"

  "280 1 false vanguard_explorerv2_eco 0.01 fedavg 1.0 10.0 1 1"
  "281 1 false vanguard_explorerv2_eco 0.01 fedprox 1.0 10.0 1 1"
  "282 1 false vanguard_explorerv2_eco 0.01 fedmax 1.0 10.0 1 1"

  "283 1 false vanguard_bladev2_balanced 0.01 fedavg 1.0 10.0 1 1"
  "284 1 false vanguard_bladev2_balanced 0.01 fedprox 1.0 10.0 1 1"
  "285 1 false vanguard_bladev2_balanced 0.01 fedmax 1.0 10.0 1 1"

  "286 1 false vanguard_bladev2_slim_a 0.01 fedavg 1.0 10.0 1 1"
  "287 1 false vanguard_bladev2_slim_a 0.01 fedprox 1.0 10.0 1 1"
  "288 1 false vanguard_bladev2_slim_a 0.01 fedmax 1.0 10.0 1 1"

  "289 1 false vanguard_bladev2_slim_b 0.01 fedavg 1.0 10.0 1 1"
  "290 1 false vanguard_bladev2_slim_b 0.01 fedprox 1.0 10.0 1 1"
  "291 1 false vanguard_bladev2_slim_b 0.01 fedmax 1.0 10.0 1 1"

  "292 1 false vanguard_blade_w13 0.01 fedavg 1.0 10.0 1 1"
  "293 1 false vanguard_blade_w13 0.01 fedprox 1.0 10.0 1 1"
  "294 1 false vanguard_blade_w13 0.01 fedmax 1.0 10.0 1 1"

  "295 1 false vanguard_blade_w14 0.01 fedavg 1.0 10.0 1 1"
  "296 1 false vanguard_blade_w14 0.01 fedprox 1.0 10.0 1 1"
  "297 1 false vanguard_blade_w14 0.01 fedmax 1.0 10.0 1 1"

  "298 1 false vanguard_blade_refine_dw 0.01 fedavg 1.0 10.0 1 1"
  "299 1 false vanguard_blade_refine_dw 0.01 fedprox 1.0 10.0 1 1"
  "300 1 false vanguard_blade_refine_dw 0.01 fedmax 1.0 10.0 1 1"

  "301 1 false vanguard_blade_deeplite 0.01 fedavg 1.0 10.0 1 1"
  "302 1 false vanguard_blade_deeplite 0.01 fedprox 1.0 10.0 1 1"
  "303 1 false vanguard_blade_deeplite 0.01 fedmax 1.0 10.0 1 1"

  "304 1 false vanguard_blade_headtrim 0.01 fedavg 1.0 10.0 1 1"
  "305 1 false vanguard_blade_headtrim 0.01 fedprox 1.0 10.0 1 1"
  "306 1 false vanguard_blade_headtrim 0.01 fedmax 1.0 10.0 1 1"
  
  
)

declare -a devices_configs=(
# hw_type | host | port | cuda_name
  "rpi sld-rpi-09.ece.utexas.edu 9090 cpu" #Change number of device
  "mc1 sld-mc1-09.ece.utexas.edu 9090 cpu" #Change number of device
)
############### TODO END CHANGE

verbose='false'
laptop_number='laptop_1'
cloud_port="9090"
cloud_cuda="cpu"
comm_rounds=30
num_devices=2
for experiment_config in "${experiment_configs[@]}"
do
  read -a exp_config <<< "$experiment_config"
  experiment="${exp_config[0]}"
  run="${exp_config[1]}"
  data_iid="${exp_config[2]}"

  # Per-experiment knobs; if any value is blank, fallback to defaults above.
  model_name="${exp_config[3]}"
  learning_rate="${exp_config[4]}"
  loss_type="${exp_config[5]}"
  mu="${exp_config[6]}"
  beta="${exp_config[7]}"
  rpi_local_epochs="${exp_config[8]}"
  mc1_local_epochs="${exp_config[9]}"

  [ -z "$model_name" ] && model_name="$default_model_name"
  [ -z "$learning_rate" ] && learning_rate="$default_learning_rate"
  [ -z "$loss_type" ] && loss_type="$default_loss_type"
  [ -z "$mu" ] && mu="$default_mu"
  [ -z "$beta" ] && beta="$default_beta"

  # Seeds for all runs are predetermined.
  if [ "$run" -eq 1 ]; then
    seed=2
  elif [ "$run" -eq 2 ]; then
    seed=14
  elif [ "$run" -eq 3 ]; then
    seed=26
  else
    echo "Invalid run value. Please specify 1, 2, or 3."
    exit 1
  fi

  dev_hw_types=()
  hosts=()
  ports=()
  cuda_names=()
  model_names=()
  dev_local_epochs=()
  for devs_configs in "${devices_configs[@]}"
  do
    read -a dev_config <<< "$devs_configs"
    hw_type="${dev_config[0]}"
    dev_hw_types+=("$hw_type")

    host="${dev_config[1]}"
    hosts+=("$host")

    port="${dev_config[2]}"
    ports+=("$port")

    cuda_name="${dev_config[3]}"
    cuda_names+=("$cuda_name")

    model_names+=("$model_name")
    if [ "$hw_type" = "rpi" ]; then
      dev_local_epochs+=("$rpi_local_epochs")
    elif [ "$hw_type" = "mc1" ]; then
      dev_local_epochs+=("$mc1_local_epochs")
    else
      echo "Unsupported hw_type: $hw_type. Expected rpi or mc1."
      exit 1
    fi
  done

  cloud_config_filename="cloud_cfg_exp${experiment}_run${run}.json"
  dev_config_filename="dev_cfg_exp${experiment}_run${run}.json"

  python generate_configs.py \
  --cloud_config_filename "${cloud_config_filename}" \
  --dev_config_filename "${dev_config_filename}" \
  --cloud_ip "${cloud_ip}" \
  --cloud_port "${cloud_port}" \
  --cloud_cuda "${cloud_cuda}" \
  --model_name "${model_name}" \
  --loss_func_name "${loss_func_name}" \
  --loss_type "${loss_type}" \
  --mu "${mu}" \
  --beta "${beta}" \
  --comm_rounds "${comm_rounds}" \
  --learning_rate "${learning_rate}" \
  --verbose "${verbose}" \
  --experiment "${experiment}" \
  --run "${run}" \
  --seed "${seed}" \
  --laptop_number "${laptop_number}" \
  --data_iid "${data_iid}" \
  --num_devices "${num_devices}" \
  --dev_hw_types "${dev_hw_types[*]}" \
  --hosts "${hosts[*]}" \
  --ports "${ports[*]}" \
  --cuda_names "${cuda_names[*]}" \
  --model_names "${model_names[*]}" \
  --dev_local_epochs "${dev_local_epochs[*]}"
done
