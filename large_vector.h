#include <vector>
#include <glm/glm.hpp>
#include <cstring>

// v不是成员报错，使用和using namespace std;
using namespace std;
template<class T>
class LargeVector {
private:
    vector<T> v;

public:
    size_t size() const { return v.size(); }
    const T& get(size_t index) const { return v[index]; }
    T& get(size_t index) { return v[index]; }

    LargeVector() { }
    LargeVector(const LargeVector& other) : v(other.v) { }
    void resize(const int size) {
        v.resize(size);
    }
    void clear(bool isIdentity = false) {
        fill(v.begin(), v.end(), T(0));
        if (isIdentity) {
            for (size_t i = 0; i < v.size(); i++) {
                v[i] = T(1);
            }
        }
    }
    T& operator[](int index) {
        return v[index];
    }
    const T& operator[](int index) const {
        return v[index];
    }

    friend LargeVector<glm::vec3> operator*(const LargeVector<glm::mat3>& other, const LargeVector<glm::vec3>& f);
    friend LargeVector<glm::vec3> operator*(const float f, const LargeVector<glm::vec3>& other);
    friend LargeVector<glm::vec3> operator-(const LargeVector<glm::vec3>& Va, const LargeVector<glm::vec3>& Vb);
    friend LargeVector<glm::vec3> operator*(const LargeVector<glm::vec3>& Va, const LargeVector<glm::vec3>& Vb);
    friend LargeVector<glm::vec3> operator+(const LargeVector<glm::vec3>& Va, const LargeVector<glm::vec3>& Vb);

    friend LargeVector<glm::mat3> operator*(const float f, const LargeVector<glm::mat3>& other);
    friend LargeVector<glm::mat3> operator-(const LargeVector<glm::mat3>& Va, const LargeVector<glm::mat3>& Vb);


    friend LargeVector<glm::vec3> operator/(const float f, const LargeVector<glm::vec3>& v);
    friend float dot(const LargeVector<glm::vec3>& Va, const LargeVector<glm::vec3>& Vb);
};

LargeVector<glm::vec3> operator*(const LargeVector<glm::mat3>& other, const LargeVector<glm::vec3>& v) {
    LargeVector<glm::vec3> tmp(v);
    for (size_t i = 0; i < v.size(); i++) {
        tmp.get(i) = other.get(i) * v.get(i);
    }
    return tmp;
}

LargeVector<glm::vec3> operator*(const LargeVector<glm::vec3>& other, const LargeVector<glm::vec3>& v) {
    LargeVector<glm::vec3> tmp(v);
    for (size_t i = 0; i < v.size(); i++) {
        tmp.get(i) = other.get(i) * v.get(i);
    }
    return tmp;
}


LargeVector<glm::vec3> operator*(const float f, const LargeVector<glm::vec3>& other) {
    LargeVector<glm::vec3> tmp(other);
    for (size_t i = 0; i < other.size(); i++) {
        tmp.get(i) = other.get(i) * f;
    }
    return tmp;
}
LargeVector<glm::mat3> operator*(const float f, const LargeVector<glm::mat3>& other) {
    LargeVector<glm::mat3> tmp(other);
    for (size_t i = 0; i < other.size(); i++) {
        tmp.get(i) = other.get(i) * f;
    }
    return tmp;
}
LargeVector<glm::vec3> operator-(const LargeVector<glm::vec3>& Va, const LargeVector<glm::vec3>& Vb) {
    LargeVector<glm::vec3> tmp(Va);
    for (size_t i = 0; i < Va.size(); i++) {
        tmp.get(i) = Va.get(i) - Vb.get(i);
    }
    return tmp;
}
LargeVector<glm::mat3> operator-(const LargeVector<glm::mat3>& Va, const LargeVector<glm::mat3>& Vb) {
    LargeVector<glm::mat3> tmp(Va);
    for (size_t i = 0; i < Va.size(); i++) {
        tmp.get(i) = Va.get(i) - Vb.get(i);
    }
    return tmp;
}

LargeVector<glm::vec3> operator+(const LargeVector<glm::vec3>& Va, const LargeVector<glm::vec3>& Vb) {
    LargeVector<glm::vec3> tmp(Va);
    for (size_t i = 0; i < Va.size(); i++) {
        tmp.get(i) = Va.get(i) + Vb.get(i);
    }
    return tmp;
}

LargeVector<glm::vec3> operator/(const float f, const LargeVector<glm::vec3>& v) {
    LargeVector<glm::vec3> tmp(v);
    for (size_t i = 0; i < v.size(); i++) {
        tmp.get(i) = v.get(i) / f;
    }
    return tmp;
}


float dot(const LargeVector<glm::vec3>& Va, const LargeVector<glm::vec3>& Vb) {
    float sum = 0;
    for (size_t i = 0; i < Va.size(); i++) {
        sum += glm::dot(Va.get(i), Vb.get(i));
    }
    return sum;
}