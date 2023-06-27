#include <GL/glew.h>
#include <GL/wglew.h>
#include <GL/freeglut.h>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp> //for matrices
#include <glm/gtc/type_ptr.hpp>
#include <vtk-9.2/vtkSmartPointer.h>
#include <vtk-9.2/vtkPoints.h>
#include <vtk-9.2/vtkPolyData.h>
#include <vtk-9.2/vtkXMLPolyDataWriter.h>
#include <vtk-9.2/vtkTriangle.h>
#include "large_vector.h" // 为了省了写for循环

#pragma comment(lib, "glew32.lib")

using namespace std;

// ------ View parameters ------
const int width = 1024, height = 1024;
int oldX = 0, oldY = 0;
float rX = 15, rY = 0;
int state = 1;
float dist = -23;
const int GRID_SIZE = 10;
GLint viewport[4];
GLdouble MV[16];
GLdouble P[16];
LARGE_INTEGER frequency;        // ticks per second
LARGE_INTEGER t1, t2;           // ticks
double frameTimeQP = 0;
float frameTime = 0;
int exportedFrames = 0;

glm::vec3 Up = glm::vec3(0, 1, 0), Right, viewDir;
float startTime = 0, fps = 0;
int totalFrames = 0;
int totalTime = 0;

// ------ Paricle & Spring parameters ------
struct Spring {
	int p1, p2;
	float rest_length;
	float Ks, Kd;
	int type;
};

vector<GLushort> indices;
vector<Spring> springs;

// 网格会分布在[-halfsize, halfsize] x[0, fullsize] 的区域里。
int numX = 20, numY = 20;
const size_t total_points = (numX + 1) * (numY + 1);
float fullsize = 4.0f;
float halfsize = fullsize / 2.0f;
glm::vec3 gravity = glm::vec3(0.0f, -0.0981f, 0.0f);
float mass = 1.f;

// 标记弹簧的类型
const int STRUCTURAL_SPRING = 0;
const int SHEAR_SPRING = 1;
const int BEND_SPRING = 2;
// 弹簧参数
int spring_count = 0;
const float DEFAULT_DAMPING = -0.125f;
float	KsStruct = 0.75f, KdStruct = -0.25f;
float	KsShear = 0.75f, KdShear = -0.25f;
float	KsBend = 0.95f, KdBend = -0.25f;

vector<glm::vec3> Particle;// position
vector<glm::vec3> Velocity;// velocity
vector<glm::vec3> F;// force

// ------- Collision detection， 椭球体参数 -------
glm::mat4 ellipsoid, inverse_ellipsoid;
int iStacks = 30;
int iSlices = 30;
float fRadius = 1;

glm::vec3 center = glm::vec3(0, 0, 0); //object space center of ellipsoid
float radius = 1;					 //object space radius of ellipsoid


// ------ Implicit Integration & Conjugate Gradient parameters------
LargeVector<glm::mat3> A;
LargeVector<glm::vec3> b;
glm::mat3 M = glm::mat3(1.0f);

vector<glm::vec3> dc_dp; //  dc/dp
vector<glm::mat3> df_dx; //  df/dp
vector<glm::mat3> df_dv; //  df/dv
LargeVector<glm::vec3> dV;

LargeVector<glm::vec3> P_;
LargeVector<glm::vec3> P_inv;
vector<float> inv_len;
vector<float> C; //for implicit integration
vector<float> C_Dot; //for implicit integration
vector<glm::vec3> deltaP2;//for implicit integration
	// conjugate parameters
const float EPS = 0.001f;
const float EPS2 = EPS * EPS;
const int i_max = 10;

// ------ Simulation parameters------
float timeStep = 10 / 60.0f;
float currentTime = 0;
double accumulator = timeStep;
int selected_index = -1;

char info[MAX_PATH] = { 0 };

void StepPhysics(float dt);

void SolveConjugateGradientPreconditioned(LargeVector<glm::mat3> A, LargeVector<glm::vec3>& x, LargeVector<glm::vec3> b, LargeVector<glm::vec3> P, LargeVector<glm::vec3> P_inv) {
	float i = 0;

	// 如果 r 接近于零向量，则说明我们的解已经非常接近真实解了。
	LargeVector<glm::vec3> r = (b - A * x);

	// 预处理矩阵的作用是改善系统的性质（如病态程度），使得迭代方法能够更快地收敛。
	LargeVector<glm::vec3> d = P_inv * r;
	LargeVector<glm::vec3> q;
	float alpha_new = 0;
	float alpha = 0;
	float beta = 0;
	float delta_old = 0;
	float delta_new = dot(r, P * r);
	float delta0 = delta_new;
	while (i<i_max && delta_new > EPS2 * delta0) {
		// 计算 q，这里 q = A * d
		q = A * d;

		// 计算步长 alpha，表示在当前搜索方向 d 上应该前进的长度
		alpha = delta_new / dot(d, q);

		// 更新解 x
		x = x + alpha * d;

		// 更新残差 r
		r = r - alpha * q;

		// 计算新的误差 delta_new
		delta_old = delta_new;
		delta_new = dot(r, r);

		// 计算搜索方向的调整系数 beta
		beta = delta_new / delta_old;

		// 更新搜索方向 d
		d = r + beta * d;
		i++;
	}
}

void AddSpring(int a, int b, float ks, float kd, int type) {
	Spring spring;
	spring.p1 = a;
	spring.p2 = b;
	spring.Ks = ks;
	spring.Kd = kd;
	spring.type = type;
	glm::vec3 deltaP = Particle[a] - Particle[b];
	spring.rest_length = sqrt(glm::dot(deltaP, deltaP));
	springs.push_back(spring);
}

void initMassSpring();

void InitGL() {
	startTime = (float)glutGet(GLUT_ELAPSED_TIME);
	currentTime = startTime;

	// get ticks per second
	QueryPerformanceFrequency(&frequency);

	// start timer
	QueryPerformanceCounter(&t1);

	glEnable(GL_DEPTH_TEST);

	initMassSpring();

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	//glPolygonMode(GL_BACK, GL_LINE);
	glPointSize(5);

	wglSwapIntervalEXT(0);
}

void initMassSpring() {
	int i = 0, j = 0, count = 0;
	int l1 = 0, l2 = 0;
	float ypos = 7.0f;
	int v = numY + 1;
	int u = numX + 1;

	indices.resize(numX * numY * 2 * 3);
	Particle.resize(total_points);
	Velocity.resize(total_points);
	F.resize(total_points);

	A.resize(total_points);
	b.resize(total_points);
	dV.resize(total_points);
	P_.resize(total_points);
	P_inv.resize(total_points);


	//fill in X
	for (j = 0; j <= numY; j++) {
		for (i = 0; i <= numX; i++) {
			Particle[count++] = glm::vec3(((float(i) / (u - 1)) * 2 - 1) * halfsize, fullsize + 1, ((float(j) / (v - 1)) * fullsize));
		}
	}

	//fill in V
	memset(&(Velocity[0].x), 0, total_points * sizeof(glm::vec3));

	//fill in indices
	GLushort* id = &indices[0];
	for (i = 0; i < numY; i++) {
		for (j = 0; j < numX; j++) {
			int i0 = i * (numX + 1) + j;
			int i1 = i0 + 1;
			int i2 = i0 + (numX + 1);
			int i3 = i2 + 1;
			if ((j + i) % 2) {
				*id++ = i0; *id++ = i2; *id++ = i1;
				*id++ = i1; *id++ = i2; *id++ = i3;
			}
			else {
				*id++ = i0; *id++ = i2; *id++ = i3;
				*id++ = i0; *id++ = i3; *id++ = i1;
			}
		}
	}

	//setup springs
	// Horizontal
	for (l1 = 0; l1 < v; l1++)	// v
		for (l2 = 0; l2 < (u - 1); l2++) {
			AddSpring((l1 * u) + l2, (l1 * u) + l2 + 1, KsStruct, KdStruct, STRUCTURAL_SPRING);
		}

	// Vertical
	for (l1 = 0; l1 < (u); l1++)
		for (l2 = 0; l2 < (v - 1); l2++) {
			AddSpring((l2 * u) + l1, ((l2 + 1) * u) + l1, KsStruct, KdStruct, STRUCTURAL_SPRING);
		}


	// Shearing Springs
	for (l1 = 0; l1 < (v - 1); l1++)
		for (l2 = 0; l2 < (u - 1); l2++) {
			AddSpring((l1 * u) + l2, ((l1 + 1) * u) + l2 + 1, KsShear, KdShear, SHEAR_SPRING);
			AddSpring(((l1 + 1) * u) + l2, (l1 * u) + l2 + 1, KsShear, KdShear, SHEAR_SPRING);
		}


	// Bend Springs
	for (l1 = 0; l1 < (v); l1++) {
		for (l2 = 0; l2 < (u - 2); l2++) {
			AddSpring((l1 * u) + l2, (l1 * u) + l2 + 2, KsBend, KdBend, BEND_SPRING);
		}
		AddSpring((l1 * u) + (u - 3), (l1 * u) + (u - 1), KsBend, KdBend, BEND_SPRING);
	}
	for (l1 = 0; l1 < (u); l1++) {
		for (l2 = 0; l2 < (v - 2); l2++) {
			AddSpring((l2 * u) + l1, ((l2 + 2) * u) + l1, KsBend, KdBend, BEND_SPRING);
		}
		AddSpring(((v - 3) * u) + l1, ((v - 1) * u) + l1, KsBend, KdBend, BEND_SPRING);
	}

	int total_springs = springs.size();
	C.resize(total_springs);
	inv_len.resize(total_springs);
	C_Dot.resize(total_springs);
	dc_dp.resize(total_springs);
	deltaP2.resize(total_springs);
	df_dx.resize(total_springs);
	df_dv.resize(total_springs);
	memset(&(C[0]), 0, total_springs * sizeof(float));
	memset(&(C_Dot[0]), 0, total_springs * sizeof(float));
	memset(&(deltaP2[0].x), 0, total_springs * sizeof(glm::vec3));

	memset(&(P_[0].x), 0, total_points * sizeof(glm::vec3));
	memset(&(P_inv[0].x), 0, total_points * sizeof(glm::vec3));

	//create a basic ellipsoid object
	ellipsoid = glm::translate(glm::mat4(1), glm::vec3(0, 2, 0));
	ellipsoid = glm::rotate(ellipsoid, 45.0f, glm::vec3(1, 0, 0));
	ellipsoid = glm::scale(ellipsoid, glm::vec3(fRadius, fRadius, fRadius / 2));
	inverse_ellipsoid = glm::inverse(ellipsoid);
}

void OnMouseDown(int button, int s, int x, int y)
{
	if (s == GLUT_DOWN)
	{
		oldX = x;
		oldY = y;
		int window_y = (height - y);
		float norm_y = float(window_y) / float(height / 2.0);
		int window_x = x;
		float norm_x = float(window_x) / float(width / 2.0);

		float winZ = 0;
		glReadPixels(x, height - y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &winZ);
		if (winZ == 1)
			winZ = 0;
		double objX = 0, objY = 0, objZ = 0;
		gluUnProject(window_x, window_y, winZ, MV, P, viewport, &objX, &objY, &objZ);
		glm::vec3 pt(objX, objY, objZ);
		size_t i = 0;
		for (i = 0; i < total_points; i++) {
			if (glm::distance(Particle[i], pt) < 0.1) {
				selected_index = i;
				printf("Intersected at %d\n", i);
				break;
			}
		}
	}

	if (button == GLUT_MIDDLE_BUTTON)
		state = 0;
	else
		state = 1;

	if (s == GLUT_UP) {
		selected_index = -1;
		glutSetCursor(GLUT_CURSOR_INHERIT);
	}
}

void OnMouseMove(int x, int y)
{
	if (selected_index == -1) {
		if (state == 0)
			dist *= (1 + (y - oldY) / 60.0f);
		else
		{
			rY += (x - oldX) / 5.0f;
			rX += (y - oldY) / 5.0f;
		}
	}
	else {
		float delta = 1500 / abs(dist);
		float valX = (x - oldX) / delta;
		float valY = (oldY - y) / delta;
		if (abs(valX) > abs(valY))
			glutSetCursor(GLUT_CURSOR_LEFT_RIGHT);
		else
			glutSetCursor(GLUT_CURSOR_UP_DOWN);

		Velocity[selected_index] = glm::vec3(0);
		Particle[selected_index].x += Right[0] * valX;
		float newValue = Particle[selected_index].y + Up[1] * valY;
		if (newValue > 0)
			Particle[selected_index].y = newValue;
		Particle[selected_index].z += Right[2] * valX + Up[2] * valY;
	}
	oldX = x;
	oldY = y;

	glutPostRedisplay();
}


void DrawGrid()
{
	glBegin(GL_LINES);
	glColor3f(0.5f, 0.5f, 0.5f);
	for (int i = -GRID_SIZE; i <= GRID_SIZE; i++)
	{
		glVertex3f((float)i, 0, (float)-GRID_SIZE);
		glVertex3f((float)i, 0, (float)GRID_SIZE);

		glVertex3f((float)-GRID_SIZE, 0, (float)i);
		glVertex3f((float)GRID_SIZE, 0, (float)i);
	}
	glEnd();
}



void OnReshape(int nw, int nh) {
	glViewport(0, 0, nw, nh);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60, (GLfloat)nw / (GLfloat)nh, 1.f, 100.0f);

	glGetIntegerv(GL_VIEWPORT, viewport);
	glGetDoublev(GL_PROJECTION_MATRIX, P);

	glMatrixMode(GL_MODELVIEW);
}

void OnShutdown() {
	Particle.clear();
	Velocity.clear();
	F.clear();
	indices.clear();
	springs.clear();
	dc_dp.clear();
	df_dx.clear();
	df_dv.clear();
	C.clear();
	C_Dot.clear();
	deltaP2.clear();

	dV.clear();
	A.clear();

	b.clear();
	P_.clear();
	P_inv.clear();
	inv_len.clear();
}
void ComputeForces() {
	size_t i = 0;

	// 遍历所有的点
	for (i = 0; i < total_points; i++) {
		// 初始化力为0
		F[i] = glm::vec3(0);

		// 如果点不在边缘，添加重力
		if (i != 0 && i != (numX))
			F[i] += gravity;

		// 添加阻尼力，阻尼力与速度成正比
		F[i] += DEFAULT_DAMPING * Velocity[i];
	}


	// 遍历所有的弹簧
	for (i = 0; i < springs.size(); i++) {
		// 计算弹簧两端的位置和速度
		glm::vec3 p1 = Particle[springs[i].p1];
		glm::vec3 p2 = Particle[springs[i].p2];
		glm::vec3 v1 = Velocity[springs[i].p1];
		glm::vec3 v2 = Velocity[springs[i].p2];
		glm::vec3 deltaP = p1 - p2;
		glm::vec3 deltaV = v1 - v2;

		// 计算弹簧长度和其变化率
		float dist = glm::length(deltaP);
		inv_len[i] = 1.0f / dist;// 倒数后面用到，提前计算提高效率
		// 弹簧伸长量
		C[i] = dist - springs[i].rest_length;

		// 单位向量，弹簧力的方向
		dc_dp[i] = deltaP / dist;

		// 在弹簧方向上的投影分别被减去和加上，得到的是弹簧的伸长或压缩速度。这个速度影响到弹簧的阻尼力
		C_Dot[i] = glm::dot(v1, -dc_dp[i]) + glm::dot(v2, dc_dp[i]);

		// 各个矢量分量的平方，在后续计算用到，提前计算提高效率
		deltaP2[i] = glm::vec3(deltaP.x * deltaP.x, deltaP.y * deltaP.y, deltaP.z * deltaP.z);

		// 计算弹簧力，包括弹性力和阻尼力
		float leftTerm = -springs[i].Ks * (dist - springs[i].rest_length);
		float rightTerm = springs[i].Kd * (glm::dot(deltaV, deltaP) / dist);
		glm::vec3 springForce = (leftTerm + rightTerm) * glm::normalize(deltaP);

		// 如果弹簧的两端不在边缘，加上弹簧力
		if (springs[i].p1 != 0 && springs[i].p1 != numX)
			F[springs[i].p1] += springForce;
		if (springs[i].p2 != 0 && springs[i].p2 != numX)
			F[springs[i].p2] -= springForce;
	}
}

void CalcForceDerivatives() {
	// 清除力对位置和速度的导数
	memset(&(df_dx[0]), 0, total_points * sizeof(glm::mat3));
	memset(&(df_dv[0]), 0, total_points * sizeof(glm::mat3));

	size_t i = 0;

	// 二阶导数矩阵的初始化
	glm::mat3 d2C_dp2[2][2] = { glm::mat3(1.0f),glm::mat3(1.0f),glm::mat3(1.0f),glm::mat3(1.0f) };

	// 对所有弹簧进行遍历，计算每一个弹簧的力对位置和速度的导数
	for (i = 0; i < springs.size(); i++) {
		// 基于每个弹簧计算二阶导数
		// 存储弹簧的伸长量至c1
		float c1 = C[i];
		// 弹簧力对位置的二阶导数
		d2C_dp2[0][0][0][0] = (-c1 * deltaP2[i].x + c1);
		d2C_dp2[0][0][1][1] = (-c1 * deltaP2[i].y + c1);
		d2C_dp2[0][0][2][2] = (-c1 * deltaP2[i].z + c1);

		d2C_dp2[0][1][0][0] = (c1 * deltaP2[i].x - c1);
		d2C_dp2[0][1][1][1] = (c1 * deltaP2[i].y - c1);
		d2C_dp2[0][1][2][2] = (c1 * deltaP2[i].z - c1);

		d2C_dp2[1][0] = d2C_dp2[0][1];
		d2C_dp2[1][1] = d2C_dp2[0][0];

		// dc_dp[i] = deltaP / dist; 这个值会影响到弹簧力对位置和速度的导数的计算
		glm::mat3 dp1 = glm::outerProduct(dc_dp[i], dc_dp[i]);
		glm::mat3 dp2 = glm::outerProduct(dc_dp[i], -dc_dp[i]);
		glm::mat3 dp3 = glm::outerProduct(-dc_dp[i], -dc_dp[i]);

		// 使用Hooke定律（弹簧定律）和阻尼因子计算力对位置和速度的导数
		// 弹簧力对位置的一阶导数
		df_dx[i] += -springs[i].Ks * (dp1 + (d2C_dp2[0][0] * C[i])) - springs[i].Kd * (d2C_dp2[0][0] * C_Dot[i]);
		df_dx[i] += -springs[i].Ks * (dp2 + (d2C_dp2[0][1] * C[i])) - springs[i].Kd * (d2C_dp2[0][1] * C_Dot[i]);
		df_dx[i] += -springs[i].Ks * (dp2 + (d2C_dp2[1][1] * C[i])) - springs[i].Kd * (d2C_dp2[1][1] * C_Dot[i]);

		// 弹簧力对速度的一阶导数
		df_dv[i] += -springs[i].Kd * dp1;
		df_dv[i] += -springs[i].Kd * dp2;
		df_dv[i] += -springs[i].Kd * dp3;
	}
}

void IntegrateImplicit(float deltaTime) {
	float h = deltaTime;

	// 计算力对位置和速度的导数
	CalcForceDerivatives();

	// 修正项
	float y = 0.0;//correction term
	size_t i = 0;
	for (i = 0; i < total_points; i++) {
		// 构建线性系统Ax = b，用于后续的隐式积分计算
		A[i] = M - h * (df_dv[i] + h * df_dx[i]);
		b[i] = (h * (F[i] + h * df_dx[i] * (Velocity[i] + y)));

		// 构建预条件器，用于优化线性系统的求解
		P_[i] = glm::vec3(A[i][0][0], A[i][1][1], A[i][2][2]);
		P_inv[i] = 1.0f / P_[i];// glm::vec3(1.0f/A[0][0], 1.0f/A[1][1], 1.0f/A[2][2]);
	}

	// 使用预条件共轭梯度法求解线性系统，得到速度变化量dV
	SolveConjugateGradientPreconditioned(A, dV, b, P_, P_inv);

	for (i = 0; i < total_points; i++) {
		// 使用得到的速度变化量dV更新速度和位置
		Velocity[i] += (dV[i] * deltaTime);
		Particle[i] += deltaTime * Velocity[i];

		// 如果更新后的位置的y值小于0（通常表示物体穿过地面），则将y值设为0，避免穿过地面
		if (Particle[i].y < 0) {
			Particle[i].y = 0;
		}
	}
}

// Provot的动态逆技术，用于防止弹簧过度伸展
void ApplyProvotDynamicInverse() {

	for (size_t i = 0; i < springs.size(); i++) {
		//check the current lengths of all springs
		glm::vec3 p1 = Particle[springs[i].p1];
		glm::vec3 p2 = Particle[springs[i].p2];
		glm::vec3 deltaP = p1 - p2;
		float dist = glm::length(deltaP);
		if (dist > (springs[i].rest_length * 1.01f)) {
			dist -= (springs[i].rest_length * 1.01f);
			dist /= 2.0f;
			deltaP = glm::normalize(deltaP);
			deltaP *= dist;
			if (springs[i].p1 == 0 || springs[i].p1 == numX) {
				Velocity[springs[i].p2] += deltaP;
			}
			else if (springs[i].p2 == 0 || springs[i].p2 == numX) {
				Velocity[springs[i].p1] -= deltaP;
			}
			else {
				Velocity[springs[i].p1] -= deltaP;
				Velocity[springs[i].p2] += deltaP;
			}
		}
	}
}
void EllipsoidCollision() {
	for (size_t i = 0; i < total_points; i++) {
		glm::vec4 X_0 = (inverse_ellipsoid * glm::vec4(Particle[i], 1));
		glm::vec3 delta0 = glm::vec3(X_0.x, X_0.y, X_0.z) - center;
		float distance = glm::length(delta0);
		if (distance < 1.0f) {
			delta0 = (radius - distance) * delta0 / distance;

			// Transform the delta back to original space
			glm::vec3 delta;
			glm::vec3 transformInv;
			transformInv = glm::vec3(ellipsoid[0].x, ellipsoid[1].x, ellipsoid[2].x);
			transformInv /= glm::dot(transformInv, transformInv);
			delta.x = glm::dot(delta0, transformInv);
			transformInv = glm::vec3(ellipsoid[0].y, ellipsoid[1].y, ellipsoid[2].y);
			transformInv /= glm::dot(transformInv, transformInv);
			delta.y = glm::dot(delta0, transformInv);
			transformInv = glm::vec3(ellipsoid[0].z, ellipsoid[1].z, ellipsoid[2].z);
			transformInv /= glm::dot(transformInv, transformInv);
			delta.z = glm::dot(delta0, transformInv);
			Particle[i] += delta;
			Velocity[i] = glm::vec3(0);
		}
	}
}


void StepPhysics(float dt) {
	ComputeForces();

	IntegrateImplicit(dt);
	EllipsoidCollision();
	ApplyProvotDynamicInverse();

}

void OnIdle() {
	/*
	//Semi-fixed time stepping
	if ( frameTime > 0.0 )
	{
		const float deltaTime = min( frameTime, timeStep );
		StepPhysics(deltaTime );
		frameTime -= deltaTime;
	}
	*/

	//Fixed time stepping + rendering at different fps
	if (accumulator >= timeStep)
	{
		StepPhysics(timeStep);
		accumulator -= timeStep;

		exportedFrames++;

		if (exportedFrames < 1000 && exportedFrames % 3 == 0) {
			// 导出点集到 VTK
			vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
			vtkSmartPointer<vtkCellArray> polygons = vtkSmartPointer<vtkCellArray>::New();
			vtkSmartPointer<vtkTriangle> triangle = vtkSmartPointer<vtkTriangle>::New();

			for (size_t i = 0; i < total_points; i++) {
				glm::vec3 p = Particle[i];
				points->InsertNextPoint(p.x, p.y, p.z);
			}

			for (size_t i = 0; i < indices.size(); i += 3) {
				triangle->GetPointIds()->SetId(0, indices[i]);
				triangle->GetPointIds()->SetId(1, indices[i + 1]);
				triangle->GetPointIds()->SetId(2, indices[i + 2]);
				polygons->InsertNextCell(triangle);
			}

			vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
			polydata->SetPoints(points);
			polydata->SetPolys(polygons);

			char filename[256];
			sprintf_s(filename, sizeof(filename), "output_%03d.vtp", exportedFrames);  // 填充文件名字符串

			vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
			writer->SetFileName(filename);
			writer->SetInputData(polydata);
			writer->Write();
		}
	}

	glutPostRedisplay();
}

void OnRender() {
	size_t i = 0;
	float newTime = (float)glutGet(GLUT_ELAPSED_TIME);
	frameTime = newTime - currentTime;
	currentTime = newTime;
	//accumulator += frameTime;

	//Using high res. counter
	QueryPerformanceCounter(&t2);
	// compute and print the elapsed time in millisec
	frameTimeQP = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
	t1 = t2;
	accumulator += frameTimeQP;

	++totalFrames;
	if ((newTime - startTime) > 1000)
	{
		float elapsedTime = (newTime - startTime);
		fps = (totalFrames / elapsedTime) * 1000;
		startTime = newTime;
		totalFrames = 0;
	}

	sprintf_s(info, "FPS: %3.2f, Frame time (GLUT): %3.4f msecs, Frame time (QP): %3.3f", fps, frameTime, frameTimeQP);
	glutSetWindowTitle(info);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	glTranslatef(0, 0, dist);
	glRotatef(rX, 1, 0, 0);
	glRotatef(rY, 0, 1, 0);

	glGetDoublev(GL_MODELVIEW_MATRIX, MV);
	viewDir.x = (float)-MV[2];
	viewDir.y = (float)-MV[6];
	viewDir.z = (float)-MV[10];
	Right = glm::cross(viewDir, Up);

	//draw grid
	DrawGrid();

	//draw ellipsoid
	glColor3f(0, 1, 0);
	glPushMatrix();
	glMultMatrixf(glm::value_ptr(ellipsoid));
	glutWireSphere(fRadius, iSlices, iStacks);
	glPopMatrix();

	//draw polygons
	glColor3f(1, 1, 1);
	glBegin(GL_TRIANGLES);
	for (i = 0; i < indices.size(); i += 3) {
		glm::vec3 p1 = Particle[indices[i]];
		glm::vec3 p2 = Particle[indices[i + 1]];
		glm::vec3 p3 = Particle[indices[i + 2]];
		glVertex3f(p1.x, p1.y, p1.z);
		glVertex3f(p2.x, p2.y, p2.z);
		glVertex3f(p3.x, p3.y, p3.z);
	}
	glEnd();

	//draw points

	glBegin(GL_POINTS);
	for (i = 0; i < total_points; i++) {
		glm::vec3 p = Particle[i];
		int is = (i == selected_index);
		glColor3f((float)!is, (float)is, (float)is);
		glVertex3f(p.x, p.y, p.z);
	}
	glEnd();

	glutSwapBuffers();
}

void main(int argc, char** argv) {

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize(width, height);
	glutCreateWindow("GLUT Cloth Demo [Implicit Integration-Baraff & Witkin's Model]");

	glutDisplayFunc(OnRender);
	glutReshapeFunc(OnReshape);
	glutIdleFunc(OnIdle);

	glutMouseFunc(OnMouseDown);
	glutMotionFunc(OnMouseMove);
	glutCloseFunc(OnShutdown);

	glewInit();
	InitGL();

	glutMainLoop();
}