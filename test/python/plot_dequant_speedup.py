import matplotlib.pyplot as plt
import numpy as np
import os

import samples
from profile_script import (profile_type_list, shape_list,
                            written_data_directory, written_data_label, 
                            dont_need_dequant)

plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.marker'] = 'o'
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['xtick.minor.visible'] = False
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams["legend.frameon"] = False
plt.rcParams['figure.titlesize'] = 'x-large'
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams["figure.figsize"] = (8, 6)


def bar_chart(shape_list, profile_type_list, plot_target_dict, plot_type_dict,
               ylabel, suptitle, file_label, plot_logy=False,
               show_type_label=False, show_target_label=False):
    typeid_list = [find_type_index(profile_type_list, type) for type in plot_type_dict.values()]
    assert all(idx != -1 for idx in typeid_list)
    x_tick_num = len(shape_list)
    x_tick_index = np.arange(x_tick_num)

    bar_width = 0.35
    fig = plt.figure()
    ax = fig.subplots(1, 1)
    for i, type_label in enumerate(plot_type_dict.keys()):
        bottom_accumulator = [0] * x_tick_num
        for item_num, (target_label, ydata) in enumerate(plot_target_dict.items()):
            label = None
            if show_type_label and show_target_label:
                label = type_label + "-" + target_label
            elif show_type_label:
                label = type_label
            elif show_target_label:
                label = target_label
            if not np.all(ydata[typeid_list[i]] == 0):
                ax.bar(x_tick_index + i * bar_width, ydata[typeid_list[i]], bar_width, bottom=bottom_accumulator, label=label)
            bottom_accumulator += ydata[typeid_list[i]]

    ax.set_xticks(x_tick_index + bar_width / 2, shape_list)
    ax.legend(bbox_to_anchor=(0.99, 1.03), loc='upper left')
    ax.set_xlabel("Matrix Dimension(M=N=K)")
    ax.set_ylabel(ylabel)
    fig.autofmt_xdate()
    fig.suptitle(suptitle, y=0.92)
    fig.savefig(image_directory_path + f"bar_chart_{file_label}.pdf", format="pdf", bbox_inches='tight')

def bar_chart_speedup(shape_list, profile_type_list, plot_target_dict, plot_type_dict,
               ylabel, suptitle, file_label, plot_logy=False,
               show_type_label=False, show_target_label=False):
    typeid_list = [find_type_index(profile_type_list, type) for type in plot_type_dict.values()]
    assert all(idx != -1 for idx in typeid_list)
    x_tick_num = len(shape_list)
    x_tick_index = np.arange(x_tick_num)

    bar_width = 0.35
    fig = plt.figure()
    ax = fig.subplots(1, 1)
    baseline = [0] * x_tick_num
    for target_label, ydata in plot_target_dict.items():
        baseline += ydata[typeid_list[0]]

    for i, type_label in enumerate(plot_type_dict.keys()):
        bottom_accumulator = [0] * x_tick_num
        for item_num, (target_label, ydata) in enumerate(plot_target_dict.items()):
            label = None
            if show_type_label and show_target_label:
                label = type_label + "-" + target_label
            elif show_type_label:
                label = type_label
            elif show_target_label:
                label = target_label
            part_speedup = [0] * x_tick_num
            if not np.all(ydata[typeid_list[i]] == 0):
                part_speedup = ydata[typeid_list[i]] / baseline 
                ax.bar(x_tick_index + i * bar_width, part_speedup, bar_width, bottom=bottom_accumulator, label=label)
            bottom_accumulator += part_speedup
            

    ax.set_xticks(x_tick_index + bar_width / 2, shape_list)
    ax.legend(bbox_to_anchor=(0.99, 1.03), loc='upper left')
    ax.set_xlabel("Matrix Dimension(M=N=K)")
    ax.set_ylabel(ylabel)
    fig.autofmt_xdate()
    fig.suptitle(suptitle, y=0.92)
    fig.savefig(image_directory_path + f"bar_chart_{file_label}.pdf", format="pdf", bbox_inches='tight')

def reader(data_dict, data_label, written_data_directory):
    for label in data_label:
        with open(written_data_directory + label + ".txt", "r") as f:
            data_dict[label] = np.loadtxt(f)


def find_type_index(type_list, type):
    return next(
        (i for i, type_obj in enumerate(type_list) if type_obj == type),
        -1
    )


if __name__ == "__main__":
    image_directory_path = "test/python/image/"
    if not os.path.exists(image_directory_path):
        os.makedirs(image_directory_path)

    fetched_data_dict = {}

    reader(fetched_data_dict, written_data_label, written_data_directory)

#--------------------------------------------------
    bar_chart_target_label = [
        # "total",
        "quantization", 
        "matmul", 
        "dequantization",
        # "performance",
    ]

    bar_chart_type_dict = {
        # "W4A4-4:8": samples.Int4SpMatmulInt32Out,
        # "W4A4": samples.Int4MatmulInt32Out,
        # "W8A8-2:4": samples.Int8SpmmCuspLtFp16Out,
        "W8A8": samples.Int8MatmulInt32Out,
        # "W8A8-2:4-cutlass": samples.Int8SpMatmulInt32Out,
        # "FP16": samples.FP16Matmul,
        # "FP32": samples.FP32Matmul,
        # "W4A4-fusion": samples.Int4FusionFp16Out,
        "W8A8-fusedDequantize": samples.Int8FusionFp16Out,
    }

    file_label = "W8A8"
    ylabel = "Time"
    plot_logy = False
    show_type_label = True
    show_target_label = True
    suptitle = "Performance Comparison on RTX3080"

    assert (all(item in written_data_label for item in bar_chart_target_label))

    bar_chart_target = {label: fetched_data_dict[label] for label in bar_chart_target_label}

    assert (all(item in profile_type_list for item in bar_chart_type_dict.values()))

    typeid_list = [find_type_index(profile_type_list, type) for type in bar_chart_type_dict.values()]

    bar_chart(shape_list, profile_type_list, bar_chart_target,
               bar_chart_type_dict, ylabel, suptitle,
               file_label, plot_logy, show_type_label, show_target_label)

    file_label = "speedup_W8A8"
    ylabel = "Speedup"

    bar_chart_speedup(shape_list, profile_type_list, bar_chart_target,
               bar_chart_type_dict, ylabel, suptitle,
               file_label, plot_logy, show_type_label, show_target_label)
#-------------------------------------------
    bar_chart_target_label = [
        # "total",
        "quantization", 
        "matmul", 
        "dequantization",
        # "performance",
    ]

    bar_chart_type_dict = {
        # "W4A4-4:8": samples.Int4SpMatmulInt32Out,
        "W4A4": samples.Int4MatmulInt32Out,
        # "W8A8-2:4": samples.Int8SpmmCuspLtFp16Out,
        # "W8A8": samples.Int8MatmulInt32Out,
        # "W8A8-2:4-cutlass": samples.Int8SpMatmulInt32Out,
        # "FP16": samples.FP16Matmul,
        # "FP32": samples.FP32Matmul,
        "W4A4-fusion": samples.Int4FusionFp16Out,
        # "W8A8-fusedDequantize": samples.Int8FusionFp16Out,
    }

    file_label = "W4A4"
    ylabel = "Time"
    plot_logy = False
    show_type_label = True
    show_target_label = True
    suptitle = "Performance Comparison on RTX3080"

    assert (all(item in written_data_label for item in bar_chart_target_label))

    bar_chart_target = {label: fetched_data_dict[label] for label in bar_chart_target_label}

    assert (all(item in profile_type_list for item in bar_chart_type_dict.values()))

    typeid_list = [find_type_index(profile_type_list, type) for type in bar_chart_type_dict.values()]

    bar_chart(shape_list, profile_type_list, bar_chart_target,
               bar_chart_type_dict, ylabel, suptitle,
               file_label, plot_logy, show_type_label, show_target_label)

    file_label = "speedup_W4A4"
    ylabel = "Speedup"

    bar_chart_speedup(shape_list, profile_type_list, bar_chart_target,
               bar_chart_type_dict, ylabel, suptitle,
               file_label, plot_logy, show_type_label, show_target_label)