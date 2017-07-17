
class FFT1D
{
public:
	FFT1D();
	~FFT1D();
};
class FFT2D
{
public:
	FFT2D(int width,int height);

	void exec(float *src,float *dst_re,float *dst_im);
	void inverse_exec();

	~FFT2D();

private:
	void createFactor(const int num,int &factor_size,int *factors);
	void createShuffle1DAndWave(const int length,const int factor_size,int *shuffle_1d);
	void createShuffle2DAndWave();
	void shuffle1();
	void colApply();
	void shuffle2();
	void rowApply();
	void shuffle3();
	int width;
	int height;
	int area;

	int width_2;
	int width2;
	int width3;

	int factors[68];
	int *factors_col;
	int *factors_row;
	int factors_col_size;
	int factors_row_size;
	int *shuffle_1d;
	int *shuffle_col;
	int *shuffle_row;

	int *shuffle_2d;

	__m128 *wave;
	__m128 *wave_col_re;
	__m128 *wave_col_im;
	__m128 *wave_row_re;
	__m128 *wave_row_im;

	__m128 *simd_re;
	__m128 *simd_im;
};
