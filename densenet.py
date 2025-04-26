# Implementation based on https://amaarora.github.io/posts/2020-08-02-densenets.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from tensorflow import Tensor
from typing import OrderedDict

class DenseLayer(nn.Module):
    # Defines a single layer inside of a DenseBlock
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        # first convolution layer: 1x1 convolution, bottleneck layer
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        # second convolution layer: 3x3 convolution
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = drop_rate

    def bn_function(self, inputs):
        # concatenate the input features from previous layers
        concatenated_features = torch.cat(inputs, 1)
        output = self.norm1(concatenated_features)
        output = self.relu1(output)
        output = self.conv1(output)
        return output
    
    def forward(self, input):
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        # apply the first convolution layer
        bottleneck_output = self.bn_function(prev_features)
        # apply the second convolution layer
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features
    
class DenseBlock(nn.Module):
    # Defines a block containing multiple DenseLayers
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()

        # create the DenseLayers
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate, # original input features + growth rate * number of layers
                growth_rate, # number of new feature maps to add per layer
                bn_size,
                drop_rate
            )
            self.layers.append(layer)

    def forward(self, init_features):
        # initialize the features with the input features
        features = [init_features]
        for layer in self.layers:
            # each layer takes the concatenated features from all previous layers
            new_features = layer(features)
            features.append(new_features)
        # output: concatenated features from all layers [input, layer1_output, layerN_output]
        return torch.cat(features, 1)
    
class Transition(nn.Module):
    # Transition between DenseBlocks
    def __init__(self, num_input_features, num_output_features):
        super(Transition, self).__init__()
        # normalize input features
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        # 1x1 convolution to reduce the number of feature maps
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        # average pooling to reduce dimensions
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, input):
        # batch normalization -> ReLU -> 1x1 convolution -> average pooling
        output = self.norm(input)
        output = self.relu(output)
        output = self.conv(output)
        output = self.pool(output)
        return output
    
class DenseNet(nn.Module):
    def __init__(self, growth_rate=12, block_config=(6, 12, 24), num_init_features=32, bn_size=4, drop_rate=0.2, num_classes=36, num_characters=5):
        """
        growth_rate: number of new features to add per dense layer
        block_config: number of layers in each dense block
        num_init_features: number of features before the first dense block
        bn_size: size of the bottleneck layer
        num_classes: number possible classes (a-z, 0-9)
        num_characters: number of characters in the output sequence (5 letter captchas)
        """
        super().__init__()

        # Initial convolution layer
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_init_features, 7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(3, stride=2, padding=1))
        ]))

        # Dense blocks and transitions
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            # create a DenseBlock with a specific number of layers
            block = DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
            self.features.add_module(f'denseblock{i + 1}', block)

            # add growth_rate * num_layers to the number of features
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = Transition(num_features, num_features // 2)
                self.features.add_module(f'transition{i + 1}', trans)
                num_features = num_features // 2

        # Final batch normalization
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))
        # output for each character
        self.heads = nn.ModuleList([nn.Linear(num_features, num_classes) for _ in range(num_characters)])

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        return [head(x) for head in self.heads]