import os
import numpy as np

import tvm
from tvm import autotvm
from tvm import relay
from tvm.relay import testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import tvm.contrib.graph_runtime as runtime

def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        n_layer = int(name.split('-')[1])
        net, params = relay.testing.resnet.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif "vgg" in name:
        n_layer = int(name.split('-')[1])
        net, params = relay.testing.vgg.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif name == 'mobilenet':
        net, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'squeezenet_v1.1':
        net, params = relay.testing.squeezenet.get_workload(batch_size=batch_size, version='1.1', dtype=dtype)
    elif name == 'inception_v3':
        input_shape = (1, 3, 299, 299)
        net, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'mxnet':
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model
        block = get_model('resnet18_v1', pretrained=True)
        net, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)
        net = relay.Function(net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs)
    else:
        raise ValueError("Unsupported network: " + name)

    return net, params, input_shape, output_shape

# Replace "llvm" with the correct target of your CPU.
# For example, for AWS EC2 c5 instance with Intel Xeon
# Platinum 8000 series, the target should be "llvm -mcpu=skylake-avx512".
# For AWS EC2 c4 instance with Intel Xeon E5-2666 v3, it should be
# "llvm -mcpu=core-avx2".
target = "llvm"

batch_size = 1
dtype = "float32"
model_name = "resnet-18"
log_file = "%s.log" % model_name

# Set number of threads used for tuning based on the number of
# physical CPU cores on your machine.
num_threads = 4
os.environ["TVM_NUM_THREADS"] = str(num_threads)

tuning_option = {
    'log_filename': log_file,
    'tuner': 'ga',
    'early_stopping': None,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(number=10, repeat=1,
                                   min_repeat_ms=1000),
    ),
}

# You can skip the implementation of this function for this tutorial.
def tune_kernels(tasks,
                 measure_option,
                 tuner='gridsearch',
                 early_stopping=None,
                 log_filename='tuning.log'):

    for i, tsk in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # converting conv2d tasks to conv2d_NCHWc tasks
        op_name = tsk.workload[0]
        if op_name == 'conv2d':
            func_create = 'topi_x86_conv2d_NCHWc'
        elif op_name == 'depthwise_conv2d_nchw':
            func_create = 'topi_x86_depthwise_conv2d_NCHWc_from_nchw'
        else:
            raise ValueError("Tuning {} is not supported on x86".format(op_name))

        task = autotvm.task.create(func_create, args=tsk.args,
                                   target=target, template_key='direct')
        task.workload = tsk.workload

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(task, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(task)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        n_trial=len(task.config_space)
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(log_filename)])

def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    net, params, data_shape, out_shape = get_network(model_name, batch_size)
    tasks = autotvm.task.extract_from_program(net, target=target,
                                              params=params, ops=(relay.op.nn.conv2d,))

    # run tuning tasks
    print("Tuning...")
    tune_kernels(tasks, **tuning_opt)

    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                net, target=target,  params=params)

        # upload parameters to device
        ctx = tvm.cpu()
        data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
        module = runtime.create(graph, lib, ctx)
        module.set_input('data', data_tvm)
        module.set_input(**params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=100, repeat=3)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))

# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.

tune_and_evaluate(tuning_option)
