// In the three types of integrators, the spring-mass system exhibits changes in kinetic energy, potential energy, and total energy.
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <vector>

Eigen::Vector2d simulate_one_step_explicit_euler(const Eigen::Vector2d& current, double h, double m, double k) {
    Eigen::Vector2d next;

    double a = -k * current[0] / m;

    next[0] = current[0] + h * current[1];
    next[1] = current[1] + h * a;

    return next;
}

Eigen::Vector2d simulate_one_step_implicit_euler(const Eigen::Vector2d& current, double h, double m, double k) {
    // (m + h^2 * k) * dv = - h * k * (x + h * v)
    Eigen::Vector2d next;

    double a = -k * current[0] / m;

    double dv = -h * k * (current[0] + h * current[1]) / (m + h * h * k);
    double dx = h * (current[1] + dv);

    next[0] = current[0] + dx;
    next[1] = current[1] + dv;

    return next;
}

Eigen::Vector2d simulate_one_step_semi_implicit_euler(const Eigen::Vector2d& current, double h, double m, double k) {
    Eigen::Vector2d next;

    double a = -k * current[0] / m;

    next[1] = current[1] + h * a;
    next[0] = current[0] + h * next[1];// 半隐式用的是vj+1

    return next;
}

void simulate_and_write(const std::string& filename, const Eigen::Vector2d& boundary, double h, double m, double k, Eigen::Vector2d(*simulate_one_step)(const Eigen::Vector2d&, double, double, double)) {
    Eigen::Vector2d current = boundary;

    std::ofstream file(filename);
    file << "time,kinetic_energy,potential_energy\n";

    // calculate and record the kinetic energy and potential energy
    double kinetic_energy = 0.5 * m * current[1] * current[1];
    double potential_energy = 0.5 * k * current[0] * current[0];
    file << 0 << "," << kinetic_energy << "," << potential_energy << "\n";

    for (int j = 0; j < 1000; ++j) {
        current = simulate_one_step(current, h, m, k);

        kinetic_energy = 0.5 * m * current[1] * current[1];
        potential_energy = 0.5 * k * current[0] * current[0];
        file << (j + 1) * h << "," << kinetic_energy << "," << potential_energy << "\n";
    }

    file.close();
}

int main() {
    double m = 200.0;
    double k = 50.0;
    double h = 0.1;
    Eigen::Vector2d boundary;
    boundary[0] = 5.0; // 初始点位移量为5（即弹簧伸长量为5）
    boundary[1] = 0.0; // 初试速度为0

    simulate_and_write("explicit_euler.csv", boundary, h, m, k, simulate_one_step_explicit_euler);
    simulate_and_write("implicit_euler.csv", boundary, h, m, k, simulate_one_step_implicit_euler);
    simulate_and_write("semi_implicit_euler.csv", boundary, h, m, k, simulate_one_step_semi_implicit_euler);

    return 0;
}