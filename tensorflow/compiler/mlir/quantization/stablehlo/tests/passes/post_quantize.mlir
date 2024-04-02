// RUN: stablehlo-quant-opt %s -split-input-file -stablehlo-post-quantize | FileCheck %s

// CHECK-LABEL: @remove_volatile_qdq
func.func @remove_volatile_qdq() -> tensor<3x2xf32> {
  // CHECK: %[[CST:.*]] = stablehlo.constant
  // CHECK-NOT: "quantfork.qcast"
  // CHECK-NOT: "quantfork.dcast"
  // CHECK: return %[[CST]]
  %cst = stablehlo.constant dense<[[-0.960978984, -0.390246302], [-0.790828585, -0.601039409], [-1.0280807, -1.02731466]]> : tensor<3x2xf32>
  %q = "quantfork.qcast"(%cst) {volatile} : (tensor<3x2xf32>) -> tensor<3x2x!quant.uniform<i8:f32, 0.013075299590241675:-64>>
  %dq = "quantfork.dcast"(%q) : (tensor<3x2x!quant.uniform<i8:f32, 0.013075299590241675:-64>>) -> tensor<3x2xf32>
  func.return %dq : tensor<3x2xf32>
}

// -----

// CHECK-LABEL: @remove_volatile_qdq_with_requantization
// CHECK-SAME: %[[ARG0:.*]]: tensor<3x2xf32>
func.func @remove_volatile_qdq_with_requantization(%arg0: tensor<3x2xf32>) -> tensor<3x2xf32> {
  // CHECK: %[[Q1:.*]] = stablehlo.uniform_quantize %[[ARG0]]
  // CHECK: %[[Q2:.*]] = stablehlo.uniform_quantize %[[Q1]]
  // CHECK: %[[ABS:.*]] = stablehlo.abs %[[Q2]]
  // CHECK: %[[DQ:.*]] = stablehlo.uniform_dequantize %[[ABS]]
  // CHECK: %[[ADD:.*]] = stablehlo.add %[[ARG0]], %[[DQ]]
  // CHECK: return %[[ADD]]
  %q1 = "quantfork.qcast"(%arg0) {volatile} : (tensor<3x2xf32>) -> tensor<3x2x!quant.uniform<i8:f32, 6.000000e-03:-128>>
  %q2 = "quantfork.qcast"(%q1) {volatile} : (tensor<3x2x!quant.uniform<i8:f32, 6.000000e-03:-128>>) -> tensor<3x2x!quant.uniform<i8:f32, 0.013075299590241675:-64>>
  %dq1 = "quantfork.dcast"(%q2) : (tensor<3x2x!quant.uniform<i8:f32, 0.013075299590241675:-64>>) -> tensor<3x2xf32>
  %abs = stablehlo.abs %q2 : (tensor<3x2x!quant.uniform<i8:f32, 0.013075299590241675:-64>>) -> tensor<3x2x!quant.uniform<i8:f32, 0.013075299590241675:-64>>
  %dq2 = "quantfork.dcast"(%abs) : (tensor<3x2x!quant.uniform<i8:f32, 0.013075299590241675:-64>>) -> tensor<3x2xf32>
  %add = stablehlo.add %dq1, %dq2 : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
  func.return %add : tensor<3x2xf32>
}

// -----

// CHECK-LABEL: @quantize_constant
// CHECK-SAME: %[[ARG0:.*]]: tensor<1x3xf32>
func.func @quantize_constant(%arg0: tensor<1x3xf32>) -> tensor<1x2xf32> {
  // CHECK-DAG: %[[QCST:.*]] = stablehlo.constant() {value = dense<-78> : tensor<3x2xi8>} : () -> tensor<3x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
  // CHECK-DAG: %[[Q1:.*]] = stablehlo.uniform_quantize %[[ARG0]]
  // CHECK-NOT: "quantfork.qcast"
  // CHECK: %[[DOT:.*]] = stablehlo.dot %[[Q1]], %[[QCST]]
  // CHECK: %[[DQ:.*]] = stablehlo.uniform_dequantize %[[DOT]]
  // CHECK: return %[[DQ]]
  %cst = stablehlo.constant dense<-0.390246302> : tensor<3x2xf32>
  %q1 = "quantfork.qcast"(%arg0) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 6.000000e-03:-128>>
  %q2 = "quantfork.qcast"(%cst) {volatile} : (tensor<3x2xf32>) -> tensor<3x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
  %dot = stablehlo.dot %q1, %q2 : (tensor<1x3x!quant.uniform<i8:f32, 6.000000e-03:-128>>, tensor<3x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>) -> tensor<1x2x!quant.uniform<i8:f32, 1.000000e-03:-3>>
  %dq = "quantfork.dcast"(%dot) : (tensor<1x2x!quant.uniform<i8:f32, 1.000000e-03:-3>>) -> tensor<1x2xf32>
  func.return %dq : tensor<1x2xf32>
}

// -----

// CHECK-LABEL: @convert_quantfork_qdq_to_stablehlo_uniform_qdq
// CHECK-SAME: %[[ARG0:.*]]: tensor<1x3xf32>
// CHECK-SAME: %[[ARG1:.*]]: tensor<3x2xf32>
func.func @convert_quantfork_qdq_to_stablehlo_uniform_qdq(%arg0: tensor<1x3xf32>, %arg1: tensor<3x2xf32>) -> tensor<1x2xf32> {
  // CHECK: %[[Q1:.*]] = stablehlo.uniform_quantize %[[ARG0]]
  // CHECK-NOT: "quantfork.qcast"
  // CHECK: %[[Q2:.*]] = stablehlo.uniform_quantize %[[ARG1]]
  // CHECK-NOT: "quantfork.qcast"
  // CHECK: %[[DOT:.*]] = stablehlo.dot %[[Q1]], %[[Q2]]
  // CHECK: %[[DQ:.*]] = stablehlo.uniform_dequantize %[[DOT]]
  // CHECK: return %[[DQ]]
  %q1 = "quantfork.qcast"(%arg0) {volatile} : (tensor<1x3xf32>) -> tensor<1x3x!quant.uniform<i8:f32, 6.000000e-03:-128>>
  %q2 = "quantfork.qcast"(%arg1) {volatile} : (tensor<3x2xf32>) -> tensor<3x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>
  %dot = stablehlo.dot %q1, %q2 : (tensor<1x3x!quant.uniform<i8:f32, 6.000000e-03:-128>>, tensor<3x2x!quant.uniform<i8<-127:127>:f32, 5.000000e-03>>) -> tensor<1x2x!quant.uniform<i8:f32, 1.000000e-03:-3>>
  %dq = "quantfork.dcast"(%dot) : (tensor<1x2x!quant.uniform<i8:f32, 1.000000e-03:-3>>) -> tensor<1x2xf32>
  func.return %dq : tensor<1x2xf32>
}

// -----

// Tests unquantized composite tf.XlaCallModule is converted to func.call.

module {
  // CHECK-LABEL: func.func @main
  func.func @main(%arg0: tensor<1x1024xf32>) -> tensor<1x3xf32> {
    // CHECK: call @composite_dot_general_fn_1
    // CHECK-SAME: (tensor<1x1024xf32>, tensor<1024x3xf32>) -> tensor<1x3xf32>
    // CHECK-NOT: tf.XlaCallModule
    %0 = "tf.Const"() <{value = dense<0.5> : tensor<1024x3xf32>}> : () -> tensor<1024x3xf32>
    %2 = "tf.XlaCallModule"(%arg0, %0) <{Sout = [#tf_type.shape<1x3>], dim_args_spec = [], disabled_checks = [], has_token_input_output = false, module = "", platforms = [], version = 5 : i64}> {_entry_function = @composite_dot_general_fn_1, _original_entry_function = "composite_dot_general_fn_1", _stablehlo_module_attrs = {}, _tfl_quant_trait = "fully_quantizable", device = ""} : (tensor<1x1024xf32>, tensor<1024x3xf32>) -> tensor<1x3xf32>
    return %2 : tensor<1x3xf32>
  }
  // CHECK-LABEL: func.func private @composite_dot_general_fn_1
  // CHECK-SAME: -> tensor<1x3xf32>
  func.func private @composite_dot_general_fn_1(%arg0: tensor<1x1024xf32>, %arg1: tensor<1024x3xf32>) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x1024xf32>, tensor<1024x3xf32>) -> tensor<1x3xf32>
    return %0 : tensor<1x3xf32>
  }
}
