#include <pybind11/pybind11.h>

#include "grad_check.h"
#include "history_aggr.h"
namespace py = pybind11;
using namespace pybind11::literals;

void init_grad_check(py::module &m) {
  m.def("record_history", &record_history, "");
  m.def("count_history_reconstruct", &count_history_reconstruct, "");
  m.def("count_num", &count_num, "");
  m.def("get_graph_structure_score", &get_graph_structure_score, "");
  m.def("aggr_forward_history", &aggr_forward_history, "");
  m.def("aggr_forward_history_edge_value", &aggr_forward_history_edge_value,
        "");
}

PYBIND11_MODULE(hiscache_backend, m) {
  m.doc() = "Historical cache";
  init_grad_check(m);
}