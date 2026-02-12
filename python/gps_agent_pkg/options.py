"""
Translates gps_agent_pkg/include/gps_agent_pkg/options.h.

C++ OptionsVariant is a std::variant<bool, int, double, std::vector<int>,
Eigen::VectorXd, Eigen::MatrixXd, std::string, std::vector<uint8_t>>.

In Python, any of those values can be stored as plain Python / numpy objects,
so OptionsMap is simply an alias for dict[str, Any].
"""
from __future__ import annotations
from typing import Any

# Type alias mirroring C++ OptionsMap = std::map<std::string, OptionsVariant>
OptionsMap = dict  # dict[str, Any]
