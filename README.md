# crane_mpc

First-slice ROS-free shell for W06. It installs no C++ headers, libraries, or
runtime nodes; the deterministic fixture is private to the package tests and
only exercises the installed `crane_model` mock. Acados, CasADi, model
dynamics, ROS topics, solver tuning, and actuator authority are non-goals
until the W03 model API and shadow gates are accepted. No legacy MPC or
`ax_comp_lib` is linked.
