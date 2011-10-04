
#define _USE_MATH_DEFINES // for M_PI on Visual Studio
#include <cmath>
#include <vector>
#include <complex>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace std;


typedef complex<double> Complex;
typedef vector<Complex> Vector;
typedef vector<Vector> Matrix;


const double c0 = 3.0e8;


// begin independent config params
const double sizex = 1.0e-3;
const double sizey = 1.0e-3;
const double sizez = 0.5;

const int countx = 65;
const int county = 65;
const int countz = 51;

const int countx_scale = 2;
const int county_scale = 2;
const int countz_scale = 1;

const double amplitude = 1.0;

const double wavelen = 632.8e-9;

const double beamwidth = 0.4e-3;

const double tbc_min = 1.0e-9;

const double sigma_max = 3.0e16; // max value at borders
const int sigma_power = 2; // exponent of the absorption profile
const double pml_border_size = 0.046875; // border size, relative to window size

const string output_dir = "propagator";
// end independent config params


// begin dependent config params
const double lowx = -0.5 * sizex;
const double highx = 0.5 * sizex;
const double lowy = -0.5 * sizey;
const double highy = 0.5 * sizey;
const double lowz = 0.0;
const double highz = sizez;

const double dx = sizex / (countx - 1);
const double dy = sizey / (county - 1);
const double dz = sizez / (countz - 1);
const double dx2 = dx * dx;
const double dy2 = dy * dy;
const double dz2 = dz * dz;

const double K = 2.0 * M_PI / wavelen;
const double omega = K * c0;

const double w0 = 0.5 * beamwidth;
const double w02 = w0 * w0;
// end dependent config params


// begin global numerical data
const Complex i(0.0, 1.0);

Matrix field;

Matrix field_exact;
// end global numerical data


// begin global misc data
ofstream zx_out;
ofstream zy_out;
ofstream z_out;
ofstream xy_init_out;
ofstream xy_final_out;
ofstream x_init_out;
ofstream y_init_out;
ofstream x_final_out;
ofstream y_final_out;
ofstream power_out;
// end global misc data


// begin function prototypes
void solve_analytic(int k, Matrix &fld);
// end function prototypes


void dfopen(ofstream &out, string name, string suffix)
{
	ios_base::iostate exmask = (ios_base::eofbit | ios_base::failbit | ios_base::badbit);

	out.exceptions(exmask);
	out.open((output_dir + "/" + name + "_" + suffix + ".dat").c_str());
}

void init(string suffix)
{
	field.resize(countx, Vector(county));
	field_exact.resize(countx, Vector(county));

	dfopen(zx_out, "zx_out", suffix);
	dfopen(zy_out, "zy_out", suffix);
	dfopen(z_out, "z_out", suffix);
	dfopen(xy_init_out, "xy_init_out", suffix);
	dfopen(xy_final_out, "xy_final_out", suffix);
	dfopen(x_init_out, "x_init_out", suffix);
	dfopen(y_init_out, "y_init_out", suffix);
	dfopen(x_final_out, "x_final_out", suffix);
	dfopen(y_final_out, "y_final_out", suffix);
	dfopen(power_out, "power_out", suffix);
}

void cleanup()
{
	zx_out.close();
	zy_out.close();
	z_out.close();
	xy_init_out.close();
	xy_final_out.close();
	x_init_out.close();
	y_init_out.close();
	x_final_out.close();
	y_final_out.close();
	power_out.close();
}

void set_field()
{
	// initialize field with a 2d gauss function
	solve_analytic(0, field);
}

double calc_amplitude(const Complex &c)
{
	return abs(c);
}

double calc_intensity(const Complex &c)
{
	return norm(c);
}

double calc_phase(const Complex &c)
{
	return arg(c);
}

void write_header(ostream &out, bool z, bool x, bool y)
{
	out << "# ";
	if (z)
		out << "Z ";
	if (x)
		out << "X ";
	if (y)
		out << "Y ";

	out << "amplitude intensity phase ";
	out << "(abs amplitude error) (abs intensity error) (abs phase error) ";
	out << "(rel amplitude error) (rel intensity error) (rel phase error)" << endl;
}

void write_field_point(ostream &out, bool Z, bool X, bool Y, int m, int n, int k)
{
	double x = lowx + m*dx;
	double y = lowy + n*dy;
	double z = lowz + k * dz;

	if (Z && (k % countz_scale))
		return;

	if (X && (m % countx_scale))
		return;

	if (Y && (n % county_scale))
		return;

	if (Z)
		out << z << ' ';
	if (X)
		out << x << ' ';
	if (Y)
		out << y << ' ';

	const Complex &u = field[m][n];
	double Amp = calc_amplitude(u);
	double Int = calc_intensity(u);
	double Pha = calc_phase(u);

	const Complex &eu = field_exact[m][n];
	double EAmp = calc_amplitude(eu);
	double EInt = calc_intensity(eu);
	double EPha = calc_phase(eu);

	double DAmp = abs(Amp - EAmp);
	double DInt = abs(Int - EInt);
	double DPha = abs(Pha - EPha);

	double RAmp = abs(DAmp / EAmp);
	double RInt = abs(DInt / EInt);
	double RPha = abs(DPha / EPha);

	out << Amp << ' ' << Int << ' ' << Pha << ' ';
	out << DAmp << ' ' << DInt << ' ' << DPha << ' ';
	out << RAmp << ' ' << RInt << ' ' << RPha << endl;
}

void write_field_xy(ostream &out)
{
	write_header(out, false, true, true);

	for (int m = 0; m < countx; m++)
	{
		for (int n = 0; n < county; n++)
		{
			write_field_point(out, false, true, true, m, n, -1);
		}
		if (m % countx_scale == 0)
			out << endl;
	}
}

void write_field_x(ostream &out)
{
	write_header(out, false, true, false);

	int n = county / 2;

	for (int m = 0; m < countx; m++)
	{
		write_field_point(out, false, true, false, m, n, -1);
	}
}

void write_field_y(ostream &out)
{
	write_header(out, false, false, true);

	int m = countx / 2;

	for (int n = 0; n < county; n++)
	{
		write_field_point(out, false, false, true, m, n, -1);
	}
}

void write_field_zx(ostream &out, int k, bool hdr)
{
	if (hdr)
		write_header(out, true, true, false);

	int n = county / 2;

	for (int m = 0; m < countx; m++)
	{
		write_field_point(out, true, true, false, m, n, k);
	}

	if (k % countz_scale == 0)
		out << endl;
}

void write_field_zy(ostream &out, int k, bool hdr)
{
	if (hdr)
		write_header(out, true, false, true);

	int m = countx / 2;

	for (int n = 0; n < county; n++)
	{
		write_field_point(out, true, false, true, m, n, k);
	}

	if (k % countz_scale == 0)
		out << endl;
}

void write_field_z(ostream &out, int k, bool hdr)
{
	if (hdr)
		write_header(out, true, false, false);

	int m = countx / 2;
	int n = county / 2;

	write_field_point(out, true, false, false, m, n, k);
}

void write_power(ostream &out, int k, bool hdr)
{
	// TODO: implement this
}

void solve_tridiag(int N, const Vector &a, const Vector &b, const Vector &c, const Vector &r, Vector &u)
{
	if (norm(b[0]) == 0.0)
		throw invalid_argument("b[0] is zero");

	int j;
	Complex bet;
	Vector gam(N);
	u[0] = r[0] / (bet = b[0]);
	for (j = 1; j < N; j++)
	{
		gam[j] = c[j-1] / bet;
		bet = b[j] - a[j] * gam[j];
		if (norm(bet) == 0.0)
			throw invalid_argument("bet is zero");
		u[j] = (r[j] - a[j] * u[j-1]) / bet;
	}
	for (j = (N-2); j >= 0; j--)
		u[j] -= gam[j+1] * u[j+1];
}

void set_row(Matrix &M, const Vector &V, int n)
{
	int s = M.size();
	for (int m = 0; m < s; m++)
		M[m][n] = V[m];
}

void set_col(Matrix &M, const Vector &V, int m)
{
	int s = M[m].size();
	for (int n = 0; n < s; n++)
		M[m][n] = V[n];
}

Complex calc_tbc(Complex u1, Complex u2)
{
	Complex tbc(0.0, 0.0);

	if (abs(u1) > tbc_min)
	{
		Complex k = - i * log(u2 / u1) / dx;
		if (k.real() < 0.0)
			k.real(0.0);

		tbc = exp(i * k * dx);
	}

	return tbc;
}

double calc_pml_depth(double q, double qmin, double qmax)
{
	double pml_width = (qmax - qmin) * pml_border_size;
	double qlow = qmin + pml_width;
	double qhigh = qmax - pml_width;

	double d = 0.0;

	if (q <= qlow)
		d = -(qlow - q)/pml_width;

	if (q >= qhigh)
		d = (q - qhigh)/pml_width;

	return d;
}

double calc_sigma(double q, double qmin, double qmax)
{
	double d;
	double s;

	d = calc_pml_depth(q, qmin, qmax);

	if (d == 0.0)
		return 0.0;

	s = pow(abs(d), sigma_power);

	return s * sigma_max;
}

double calc_sigma_derivative(double q, double qmin, double qmax)
{
	double pml_width = (qmax - qmin) * pml_border_size;

	if (sigma_power == 0)
	{
		return 0.0;
	}
	else
	{
		double d;
		double s;

		d = calc_pml_depth(q, qmin, qmax);

		if (d == 0.0)
			return 0.0;

		if (sigma_power == 1)
			s = d / abs(d);
		else
			s = pow(d, sigma_power - 1);

		return s * sigma_max * sigma_power / pml_width;
	}
}

void solve_analytic(int k, Matrix &fld)
{
	double z = lowz + k * dz;
	double z2 = z * z;

	double zR = M_PI * w02 / wavelen;
	double zR2 = zR * zR;

	double wz = w0 * sqrt(1 + z2 / zR2);
	double wz2 = wz * wz;

	//double Rz = z * (1 + zR2 / z2);
	double Rz_inv = z / (z2 + zR2);

	double zeta = atan(z / zR);

	for (int m = 0; m < countx; m++)
	{
		double x = lowx + m * dx;
		double x2 = x * x;

		for (int n = 0; n < county; n++)
		{
			double y = lowy + n * dy;
			double y2 = y * y;

			double r2 = x2 + y2;

			Complex exparg;
			Complex u;

			exparg = 0.0;
			exparg += -(r2 / wz2);
			//exparg += i * K * z;
			exparg += (i * K * r2 * 0.5 * Rz_inv);
			exparg += -(i * zeta);

			u = amplitude * (w0 / wz) * exp(exparg);

			fld[m][n] = u;
		}
	}
}

void step_analytic(int k)
{
	solve_analytic(k, field);
}

void step_reflect(int k)
{
	int cells = max(countx, county);
	static Matrix tmp_field(countx, Vector(county));
	static Vector diaga(cells);
	static Vector diagb(cells);
	static Vector diagc(cells);
	static Vector resvec(cells);
	static Vector U(cells);

	static Complex A = i / (2.0 * K);
	static Complex Ax = A / dx2;
	static Complex Ay = A / dy2;
	static Complex B = 2.0 / dz;

	for (int n = 0; n < county; n++)
	{
		for (int m = 0; m < countx; m++)
		{
			diaga[m] = - Ax;
			diagb[m] = B + 2.0 * Ax;
			diagc[m] = - Ax;

			resvec[m] = Complex(0.0, 0.0);

			Complex r;

			if (n > 0)
			{
				r = + Ay;
				resvec[m] = r * field[m][n-1];
			}

			r = B - 2.0 * Ay;
			resvec[m] += r * field[m][n];

			if (n < (county - 1))
			{
				r = + Ay;
				resvec[m] += r * field[m][n+1];
			}
		}

		solve_tridiag(countx, diaga, diagb, diagc, resvec, U);
		set_row(tmp_field, U, n);
	}

	for (int m = 0; m < countx; m++)
	{
		for (int n = 0; n < county; n++)
		{
			diaga[n] = - Ay;
			diagb[n] = B + 2.0 * Ay;
			diagc[n] = - Ay;

			resvec[n] = Complex(0.0, 0.0);

			Complex r;

			if (m > 0)
			{
				r = + Ax;
				resvec[n] = r * tmp_field[m-1][n];
			}

			r = B - 2.0 * Ax;
			resvec[n] += r * tmp_field[m][n];

			if (m < (countx - 1))
			{
				r = + Ax;
				resvec[n] += r * tmp_field[m+1][n];
			}
		}

		solve_tridiag(county, diaga, diagb, diagc, resvec, U);
		set_col(field, U, m);
	}
}

void step_tbc(int k)
{
	int cells = max(countx, county);
	static Matrix tmp_field(countx, Vector(county));
	static Vector diaga(cells);
	static Vector diagb(cells);
	static Vector diagc(cells);
	static Vector resvec(cells);
	static Vector U(cells);

	static Complex A = i / (2.0 * K);
	static Complex Ax = A / dx2;
	static Complex Ay = A / dy2;
	static Complex B = 2.0 / dz;

	for (int n = 0; n < county; n++)
	{
		Complex tbc_x_low = calc_tbc(field[1][n], field[0][n]);
		Complex tbc_x_high = calc_tbc(field[countx-2][n], field[countx-1][n]);

		for (int m = 0; m < countx; m++)
		{
			diaga[m] = - Ax;

			diagb[m] = B + 2.0 * Ax;
			if (m == 0)
				diagb[m] -= Ax * tbc_x_low;
			if (m == (countx - 1))
				diagb[m] -= Ax * tbc_x_high;

			diagc[m] = - Ax;

			resvec[m] = Complex(0.0, 0.0);

			Complex r;

			if (n > 0)
			{
				r = + Ay;
				resvec[m] = r * field[m][n-1];
			}

			r = B - 2.0 * Ay;
			if (n == 0)
			{
				Complex tbc_y_low = calc_tbc(field[m][1], field[m][0]);
				r += Ay * tbc_y_low;
			}
			if (n == (county - 1))
			{
				Complex tbc_y_high = calc_tbc(field[m][county-2], field[m][county-1]);
				r += Ay * tbc_y_high;
			}
			resvec[m] += r * field[m][n];

			if (n < (county - 1))
			{
				r = + Ay;
				resvec[m] += r * field[m][n+1];
			}
		}

		solve_tridiag(countx, diaga, diagb, diagc, resvec, U);
		set_row(tmp_field, U, n);
	}

	for (int m = 0; m < countx; m++)
	{
		Complex tbc_y_low = calc_tbc(tmp_field[m][1], tmp_field[m][0]);
		Complex tbc_y_high = calc_tbc(tmp_field[m][county-2], tmp_field[m][county-1]);

		for (int n = 0; n < county; n++)
		{
			diaga[n] = - Ay;

			diagb[n] = B + 2.0 * Ay;
			if (n == 0)
				diagb[n] -= Ay * tbc_y_low;
			if (n == (county - 1))
				diagb[n] -= Ay * tbc_y_high;

			diagc[n] = - Ay;

			resvec[n] = Complex(0.0, 0.0);

			Complex r;

			if (m > 0)
			{
				r = + Ax;
				resvec[n] = r * tmp_field[m-1][n];
			}

			r = B - 2.0 * Ax;
			if (m == 0)
			{
				Complex tbc_x_low = calc_tbc(tmp_field[1][n], tmp_field[0][n]);
				r += Ax * tbc_x_low;
			}
			if (m == (countx - 1))
			{
				Complex tbc_x_high = calc_tbc(tmp_field[countx-2][n], tmp_field[countx-1][n]);
				r += Ax * tbc_x_high;
			}
			resvec[n] += r * tmp_field[m][n];

			if (m < (countx - 1))
			{
				r = + Ax;
				resvec[n] += r * tmp_field[m+1][n];
			}
		}

		solve_tridiag(county, diaga, diagb, diagc, resvec, U);
		set_col(field, U, m);
	}
}

void step_pml_analytic(int k)
{
	int cells = max(countx, county);
	static Matrix tmp_field(countx, Vector(county));
	static Vector diaga(cells);
	static Vector diagb(cells);
	static Vector diagc(cells);
	static Vector resvec(cells);
	static Vector U(cells);

	static Complex A = i / (2.0 * K);
	static Complex B = 1.0 / (2.0 * K);
	static Complex C = 2.0 / dz;

	for (int n = 0; n < county; n++)
	{
		double sigmay = calc_sigma(lowy + n * dy, lowy, highy);
		double dsdy = calc_sigma_derivative(lowy + n * dy, lowy, highy) / omega;

		Complex muly = 1.0 + i * sigmay / omega;
		Complex muly2 = muly * muly;
		Complex muly3 = muly * muly * muly;

		for (int m = 0; m < countx; m++)
		{
			double sigmax = calc_sigma(lowx + m * dx, lowx, highx);
			double dsdx = calc_sigma_derivative(lowx + m * dx, lowx, highx) / omega;

			Complex mulx = 1.0 + i * sigmax / omega;
			Complex mulx2 = mulx * mulx;
			Complex mulx3 = mulx * mulx * mulx;

			diaga[m] = - A / (dx2 * mulx2) + B * dsdx / (mulx3 * 2.0 * dx);
			diagb[m] = C + 2.0 * A / (dx2 * mulx2);
			diagc[m] = - A / (dx2 * mulx2) - B * dsdx / (mulx3 * 2.0 * dx);

			resvec[m] = Complex(0.0, 0.0);

			Complex c;

			if (n > 0)
			{
				c = + A / (dy2 * muly2) - B * dsdy / (muly3 * 2.0 * dy);
				resvec[m] = c * field[m][n-1];
			}

			c = C - 2.0 * A / (dy2 * muly2);
			resvec[m] += c * field[m][n];

			if (n < (county - 1))
			{
				c = + A / (dy2 * muly2) + B * dsdy / (muly3 * 2.0 * dy);
				resvec[m] += c * field[m][n+1];
			}
		}

		solve_tridiag(countx, diaga, diagb, diagc, resvec, U);
		set_row(tmp_field, U, n);
	}

	for (int m = 0; m < countx; m++)
	{
		double sigmax = calc_sigma(lowx + m * dx, lowx, highx);
		double dsdx = calc_sigma_derivative(lowx + m * dx, lowx, highx) / omega;

		Complex mulx = 1.0 + i * sigmax / omega;
		Complex mulx2 = mulx * mulx;
		Complex mulx3 = mulx * mulx * mulx;

		for (int n = 0; n < county; n++)
		{
			double sigmay = calc_sigma(lowy + n * dy, lowy, highy);
			double dsdy = calc_sigma_derivative(lowy + n * dy, lowy, highy) / omega;

			Complex muly = 1.0 + i * sigmay / omega;
			Complex muly2 = muly * muly;
			Complex muly3 = muly * muly * muly;

			diaga[n] = - A / (dy2 * muly2) + B * dsdy / (muly3 * 2.0 * dy);
			diagb[n] = C + 2.0 * A / (dy2 * muly2);
			diagc[n] = - A / (dy2 * muly2) - B * dsdy / (muly3 * 2.0 * dy);

			resvec[n] = Complex(0.0, 0.0);

			Complex c;

			if (m > 0)
			{
				c = + A / (dx2 * mulx2) - B * dsdx / (mulx3 * 2.0 * dx);
				resvec[n] = c * tmp_field[m-1][n];
			}

			c = C - 2.0 * A / (dx2 * mulx2);
			resvec[n] += c * tmp_field[m][n];

			if (m < (countx - 1))
			{
				c = + A / (dx2 * mulx2) + B * dsdx / (mulx3 * 2.0 * dx);
				resvec[n] += c * tmp_field[m+1][n];
			}
		}

		solve_tridiag(county, diaga, diagb, diagc, resvec, U);
		set_col(field, U, m);
	}
}

void step_pml_discrete(int k)
{
	int cells = max(countx, county);
	static Matrix tmp_field(countx, Vector(county));
	static Vector diaga(cells);
	static Vector diagb(cells);
	static Vector diagc(cells);
	static Vector resvec(cells);
	static Vector U(cells);

	static Complex A = i / (2.0 * K);
	static Complex Ax = A / dx2;
	static Complex Ay = A / dy2;
	static Complex B = 2.0 / dz;

	for (int n = 0; n < county; n++)
	{
		double sigmay = calc_sigma(lowy + n * dy, lowy, highy);
		double sigmay_pre = calc_sigma(lowy + (n - 0.5) * dy, lowy, highy);
		double sigmay_post = calc_sigma(lowy + (n + 0.5) * dy, lowy, highy);

		Complex d = 1.0 / ((1.0 + i * sigmay / omega) * (1.0 + i * sigmay_pre / omega));
		Complex f = 1.0 / ((1.0 + i * sigmay / omega) * (1.0 + i * sigmay_post / omega));
		Complex e = -(d + f);

		for (int m = 0; m < countx; m++)
		{
			double sigmax = calc_sigma(lowx + m * dx, lowx, highx);
			double sigmax_pre = calc_sigma(lowx + (m - 0.5) * dx, lowx, highx);
			double sigmax_post = calc_sigma(lowx + (m + 0.5) * dx, lowx, highx);

			Complex a = 1.0 / ((1.0 + i * sigmax / omega) * (1.0 + i * sigmax_pre / omega));
			Complex c = 1.0 / ((1.0 + i * sigmax / omega) * (1.0 + i * sigmax_post / omega));
			Complex b = -(a + c);

			diaga[m] = - Ax * a;
			diagb[m] = B - Ax * b;
			diagc[m] = - Ax * c;

			resvec[m] = Complex(0.0, 0.0);

			Complex r;

			if (n > 0)
			{
				r = + Ay * d;
				resvec[m] = r * field[m][n-1];
			}

			r = B + Ay * e;
			resvec[m] += r * field[m][n];

			if (n < (county - 1))
			{
				r = + Ay * f;
				resvec[m] += r * field[m][n+1];
			}
		}

		solve_tridiag(countx, diaga, diagb, diagc, resvec, U);
		set_row(tmp_field, U, n);
	}

	for (int m = 0; m < countx; m++)
	{
		double sigmax = calc_sigma(lowx + m * dx, lowx, highx);
		double sigmax_pre = calc_sigma(lowx + (m - 0.5) * dx, lowx, highx);
		double sigmax_post = calc_sigma(lowx + (m + 0.5) * dx, lowx, highx);

		Complex a = 1.0 / ((1.0 + i * sigmax / omega) * (1.0 + i * sigmax_pre / omega));
		Complex c = 1.0 / ((1.0 + i * sigmax / omega) * (1.0 + i * sigmax_post / omega));
		Complex b = -(a + c);

		for (int n = 0; n < county; n++)
		{
			double sigmay = calc_sigma(lowy + n * dy, lowy, highy);
			double sigmay_pre = calc_sigma(lowy + (n - 0.5) * dy, lowy, highy);
			double sigmay_post = calc_sigma(lowy + (n + 0.5) * dy, lowy, highy);

			Complex d = 1.0 / ((1.0 + i * sigmay / omega) * (1.0 + i * sigmay_pre / omega));
			Complex f = 1.0 / ((1.0 + i * sigmay / omega) * (1.0 + i * sigmay_post / omega));
			Complex e = -(d + f);

			diaga[n] = - Ay * d;
			diagb[n] = B - Ay * e;
			diagc[n] = - Ay * f;

			resvec[n] = Complex(0.0, 0.0);

			Complex r;

			if (m > 0)
			{
				r = + Ax * a;
				resvec[n] = r * tmp_field[m-1][n];
			}

			r = B + Ax * b;
			resvec[n] += r * tmp_field[m][n];

			if (m < (countx - 1))
			{
				r = + Ax * c;
				resvec[n] += r * tmp_field[m+1][n];
			}
		}

		solve_tridiag(county, diaga, diagb, diagc, resvec, U);
		set_col(field, U, m);
	}
}

void propagate(void (*step_func)(int k), string suffix)
{
	init(suffix);
	set_field();

	//solve_analytic(0, field_exact);

	//write_field_xy(xy_init_out);
	//write_field_x(x_init_out);
	//write_field_y(y_init_out);
	//write_field_zx(zx_out, 0, true);
	//write_field_zy(zy_out, 0, true);
	//write_field_z(z_out, 0, true);
	//write_power(power_out, 0, true);

	for (int k = 1; k < countz; k++)
	{
		step_func(k);

		solve_analytic(k, field_exact);

		//write_field_zx(zx_out, k, false);
		//write_field_zy(zy_out, k, false);
		//write_field_z(z_out, k, false);
		//write_power(power_out, k, false);
	}

	//write_field_xy(xy_final_out);
	//write_field_x(x_final_out);
	//write_field_y(y_final_out);

	cleanup();
}

int main()
{
	for (int i = 0; i < 10; i++)
	{
		propagate(step_analytic, "analytic");
		propagate(step_reflect, "reflect");
	}
	return 0;
}
