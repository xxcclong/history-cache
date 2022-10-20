#include <pybind11/pybind11.h>

#include "grad_check.h"
namespace py = pybind11;
using namespace pybind11::literals;

void init_grad_check(py::module &m) {
  m.def("record_history", &record_history, "");
  m.def("count_history_reconstruct", &count_history_reconstruct, "");
  m.def("get_graph_structure_score", &get_graph_structure_score, "");
}

PYBIND11_MODULE(hiscache_backend, m) {
  m.doc() = "Historical cache";
  init_grad_check(m);
}
