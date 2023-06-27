// Using Implicit Euler to slove the Spring Chain 
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <fstream>

// ����ʽŷ��������һ������
Eigen::VectorXd simulate_one_step(const Eigen::VectorXd& current_offset, const Eigen::VectorXd& current_v, double h, const Eigen::MatrixXd& M, const Eigen::MatrixXd& K) {
    Eigen::VectorXd dv;
    Eigen::MatrixXd A = M + h * h * K;
    dv = A.inverse() * (-h * K * (current_offset + h * current_v));

    //std::cout << "A^(-1) = " << std::endl << A.inverse() << std::endl;
    //std::cout << "dv = " << std::endl << dv << std::endl;

    return dv;
}

int LM(int a, int e, int nel) {
    if (a == 1) {
        return e;
    }
    else if (a == 2 && e != nel) {
        return e + 1;
    }
    else {
        return -1;  // ��ֹ
    }
}

int main() {
    // ���ýڵ���
    int num_nodes = 4;
    double m_value = 200.0;  // ����
    double k_value = 50.0;  // �ն�
    double h = 0.1;         // ʱ�䲽��

    // ��ʼλ�ú��ٶ�
    Eigen::VectorXd current_offset = Eigen::VectorXd::Zero(num_nodes);
    for (int i = 1; i < num_nodes; ++i) {
        current_offset[i] = 5.0;
    }

    Eigen::VectorXd current_v = Eigen::VectorXd::Zero(num_nodes);

    // �����նȾ������������
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(num_nodes, num_nodes);
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(num_nodes, num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        M(i, i) = m_value;
    }

    Eigen::Matrix2d small_K;
    small_K << k_value, -k_value, -k_value, k_value;

    for (int e = 0; e < num_nodes - 1; ++e) {
        for (int a = 0; a < 2; ++a) {
            int i = LM(a + 1, e, num_nodes - 1);
            if (i < 0) continue;
            for (int b = 0; b < 2; ++b) {
                int j = LM(b + 1, e, num_nodes - 1);
                if (j < 0) continue;
                K(i, j) += small_K(a, b);
            }
        }
    }

    std::cout << "K = " << std::endl << K << std::endl;
    std::cout << "M = " << std::endl << M << std::endl;

    // �����������������е�λ�ú��ٶ���Ϣ
    std::vector<Eigen::VectorXd> offsets;
    std::vector<Eigen::VectorXd> velocities;
    offsets.push_back(current_offset);
    velocities.push_back(current_v);

    std::ofstream file("implicit_euler_chain_spring.csv");
    file << "time,kinetic_energy,potential_energy\n";

    // calculate and record the kinetic energy and potential energy
    double kinetic_energy = 0.5 * current_v.transpose() * M * current_v;
    double potential_energy = 0.5 * current_offset.transpose() * K * current_offset;
    file << 0 << "," << kinetic_energy << "," << potential_energy << "\n";

    // ��ʼģ��
    for (int j = 0; j < 100; ++j) {
        Eigen::VectorXd dv = simulate_one_step(current_offset, current_v, h, M, K);
        current_v += dv;
        current_offset += h * current_v;
        offsets.push_back(current_offset);
        velocities.push_back(current_v);

        double kinetic_energy = 0.5 * current_v.transpose() * M * current_v;
        double potential_energy = 0.5 * current_offset.transpose() * K * current_offset;
        file << (j + 1) * h << "," << kinetic_energy << "," << potential_energy << "\n";
    }

    file.close();

    return 0;
}