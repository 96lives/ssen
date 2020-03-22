/*  Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 *  Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
 *  Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
 *  of the code.
 */
#include "common.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PyCoordsKey::PyCoordsKey(int dim) {
  reset();
  setDimension(dim);
}

void PyCoordsKey::setTensorStride(const std::vector<int> &tensor_strides) {
  int D = getDimension();
  ASSERT(D < 0 or (D > 0 and D == tensor_strides.size()),
         "The tensor strides dimension mismatch: ", ArrToString(tensor_strides),
         ", dimension of the key: ", D);
  tensor_strides_ = tensor_strides;
}

void PyCoordsKey::stride(const std::vector<int> &strides) {
  ASSERT(getDimension() == strides.size(),
         "The size of strides: ", ArrToString(strides),
         " does not match the dimension of the PyCoordKey coordinate system: ",
         std::to_string(getDimension()), ".");
  for (int i = 0; i < getDimension(); i++)
    tensor_strides_[i] *= strides[i];
}

void PyCoordsKey::up_stride(const std::vector<int> &strides) {
  ASSERT(getDimension() == strides.size(),
         "The size of strides: ", ArrToString(strides),
         " does not match the dimension of the PyCoordKey coordinate system: ",
         std::to_string(getDimension()), ".");
  ASSERT(tensor_strides_.size() == strides.size(),
         "The size of the strides: ", ArrToString(strides),
         " does not match the size of the PyCoordKey tensor_strides_: ",
         ArrToString(tensor_strides_), ".");
  for (int i = 0; i < getDimension(); i++) {
    ASSERT(tensor_strides_[i] % strides[i] == 0,
           "The output tensor stride is not divisible by ",
           "up_strides. tensor stride: ", ArrToString(tensor_strides_),
           ", up_strides: ", ArrToString(strides), ".");
    tensor_strides_[i] /= strides[i];
  }
}

void PyCoordsKey::copy(py::object py_other) {
  PyCoordsKey *p_other = py_other.cast<PyCoordsKey *>();
  setKey(p_other->key_); // Call first to set the key_set.

  setDimension(p_other->D_);
  ASSERT(getDimension() == p_other->tensor_strides_.size(),
         "The size of strides: ", ArrToString(p_other->tensor_strides_),
         " does not match the dimension of the PyCoordKey coordinate system: ",
         std::to_string(getDimension()), ".");
  tensor_strides_ = p_other->tensor_strides_;
}

void PyCoordsKey::reset() {
  key_ = 0;
  D_ = -1;
  key_set = false;
  tensor_strides_.clear();
}

void PyCoordsKey::setKey(uint64_t key) {
  key_ = key;
  key_set = true;
}

void PyCoordsKey::setDimension(int dim) {
  ASSERT(dim > 0, "The dimension should be a positive integer, you put: ",
         std::to_string(dim), ".");
  D_ = dim;
  tensor_strides_.resize(D_);
}

uint64_t PyCoordsKey::getKey() {
  ASSERT(key_set, "PyCoordsKey: Key Not set")
  return key_;
}

uint64_t PyCoordsKey::getDimension() {
  return D_;
}

std::string PyCoordsKey::toString() const {
  return "< CoordsKey, key: " + std::to_string(key_) +
         ", tensor_stride: " + ArrToString(tensor_strides_) +
         " in dimension: " + std::to_string(D_) + "> ";
}
