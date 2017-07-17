
static const float sin_120 =  0.866025403784438646763723170752;
static const float fft5_2  =  0.559016994374947424102293417182819;
static const float fft5_3  = -0.951056516295153572116439333379382;
static const float fft5_4  = -1.538841768587626701285145288018455;
static const float fft5_5  =  0.363271264002680442947733378740309;

FFT2D::FFT2D(int _width,int _height)
	:width(_width),height(_height),area(width*height),width_2(width/2),width2(width*2),width3(width2+width)
{

	int len_2 = width+ height;

	shuffle_1d = (int *)malloc(len_2*sizeof(int));
	shuffle_col = shuffle_1d;
	shuffle_row = shuffle_1d + width;
	shuffle_2d = (int *)malloc(area*sizeof(int));
	factors_col = factors;
	factors_row = factors + 34;
	createFactor(width,factor_col_size,factors_col);
	createFactor(height,factor_row_size,factors_row);

	createShuffle2D();

	wave_col = (__m128 *)malloc((width+height)*sizeof(__m128)*2);
	wave_col_re = wave_col;
	wave_col_im = wave_col_re + width;
	wave_row_re = wave_col_im + width;
	wave_row_im = wave_row_re + height;

	createWave();
//	simd_re = (__m128 *)malloc(area*sizeof(float));
	simd_im = (__m128 *)malloc(area*sizeof(float));

}
void FFT2D::shuffle1()
{
	float *src_ptr = (float *)simd_re;
	float *shuffle_2d_ptr = shuffle_2d;
	float temp;
	for(int i=0;i<height;i+=2)
	{
		for(int j=0;j<width;j++)
		{
			float *this_ptr=src+*shuffle_2d_ptr;
			temp =*src_ptr;
			*src_ptr=*this_ptr;
			*this_ptr = temp;

			src_ptr++;
			shuffle_2d_ptr++;
		}
	}
}
void FFT2D::Radix4AddSubOperator0(__m128 *simd_re_ptr,__m128 *simd_im_ptr,const int step)
{
	__m128 *v0_re = simd_re_ptr;
        __m128 *v1_re = v0_re + step;
        __m128 *v2_re = v1_re + step;
        __m128 *v3_re = v2_re + step;

        __m128 *v0_im = simd_im_ptr;
        __m128 *v1_im = v0_im   + step;
        __m128 *v2_im = v1_im   + step;
        __m128 *v3_im = v3_im   + step;

	__m128 r0,r1,r2,r3,i0,i1,i2,i3;

	r0 = *v2_re;	i0 = *v2_im;
	r4 = *v3_re;	i4 = *v3_im;

	r1 = r0 + r4;	ii = i0 + i4;
	r3 = i0 - i4;	i3 = r4 - r0;

	r2 = *v0_re;	i2 = *v0_im;
	r4 = *v1_re;	i4 = *v1_im;

	r0 = r2 + r4;	i0 = i2 + i4;
	r2 = r2 - r4;	i2 = i2 - i4;
	
	*v0_re =  r0 + r1;	*v0_im = i0 + i1;
	*v2_re =  r0 - r1;	*v2_im = i0 - i1;
	*v1_re =  r2 + r3;	*v1_im = i2 + i3;
	*v3_re =  r2 - r3;	*v3_im = i2 - i3;
}
void FFT2D::Radix4WaveOperator0(int k,__m128 *simd_re_ptr,__m128 *simd_im_ptr,int step,int wave_index0,__m128 *wave_re,__m128 *wave_im)
{
	__m128 *v0_re = simd_re_ptr + k;
	__m128 *v1_re = v0_re + step;
	__m128 *v2_re = v1_re + step;
	__m128 *v3_re = v2_re + step;

	__m128 *v0_im = simd_im_ptr + k;
	__m128 *v1_im = v0_im + step;
	__m128 *v2_im = v1_im + step;
	__m128 *v3_re = v2_im + step;

	__m128 *w0_re = wave_re + wave_index0;
	__m128 *w1_re = w0_re   + wave_index0;
	__m128 *w2_re = w1_re   + wave_index0;

	__m128 r0,r1,r2,r3,i0,i1,i2,i3,r4,i4;

	r2 = *v1_re * *w1_re - *v1_im * *w1_im;
	i2 = *v1_re * *w2_im + *v1_im * *w1_im;
	r0 = *v2_re * *w0_im + *v2_im * *w0_re;
	i0 = *v2_re * *w0_re - *v2_im * *w0_im;
	r3 = *v3_re * *w3_im + *v3_im * *w3_re;
	i3 = *v3_re * *w3_re - *v3_im * *w3_im;

	r1 = i0 + i3;	i1 = r0 + r3;
	r3 = r0 - r3;	i3 = i3 - i0;
	r4 = *v0_re;	i4 = *v0_im;

	r0 = r4 + r2;	i0 = i4 + i2;
	r2 = r4 - r2;	i2 = i4 - i2;

	*v0_re = r0 + r1;	*v0_im = i0 + i1;
	*v2_re = r0 - r1;	*v2_im = i0 - i1;
	*v1_re = r2 + r3;	*v1_im = i2 + i3;
	*v3_re = r2 - r3;	*v3_im = i2 - i3;
}
void FFT2D::Radix2AddSubOperator0(__m128 *simd_re_ptr,__m128 *simd_im_ptr, int step)
{
	__m128 *v0_re = simd_re_ptr;
	__m128 *v1_re = v0_re + step;
	__m128 *v0_im = simd_im_ptr;
	__m128 *v1_im = v0_im + step;

	__m128 r0,i0,r1,i1;

	r0 = *v0_re + *v1_re;
	i0 = *v0_im + *v1_im;

	r1 = *v0_re - *v0_re;
	i1 = *v0_im - *v0_im;

	*v0_re = r0;	*v0_im = i0;
	*v1_re = r1;	*v1_im = i1;
}
void FFT2D::Radix2WaveOperator0(int k,__m128 *simd_re_ptr,__m128 *simd_im_ptr,int step,int wave_index0,__m128 *wave_re,__m128 *wave_im)
{
	__m128 *v0_re = simd_re_ptr + k;
	__m128 *v1_re = v0_re + step;
	__m128 *v2_re = v1_re +ｓ
	__m128 *v0_im = simd_im_ptr + k;
	__ml28 *v1_im = v0_im + step;

	__m128 *w0_re = wave_re + wave_index0;
	__m128 *w0_im = wave_im + wave_index0;

	__m128 r0,i0,r1,i1;

	r1 = *v1_re * *w0_re - *v1_im * *w0_im;
	i1 = *v1_im * *w0_re + *v1_re * *w0_im;
	r0 = *v0_re;
	i0 = *v0_im;
	
	*v0_re = r0 + r1;	*v0_im = i0 + i1;
	*v1_re = r0 - r1;	*v1_im = i0 - i1;
}

void FFT2D::Radix3AddSubOperator0(__m128 *simd_re_ptr,__m128 *simd_im_ptr, int step)
{

        __m128 *v0_re = simd_re_ptr;
        __m128 *v1_re = v0_re + step;
	__m128 *v2_re = v1_re + step;

        __m128 *v0_im = simd_im_ptr;
        __m128 *v1_im = v0_im + step;
	__m128 *v2_im = v1_im + step;

        __m128 r0,i0,r1,i1,r2,i2;

        r1 = *v1_re + *v2_re;
        i1 = *v1_im + *v2_im;
        r0 = *v0_re;
        i0 = *v0_im;

        r2 = sin_120*(*v1_im - *v2_im);
        i2 = sin_120*(*v2_re - *v1_re);
        *v0_re = r0 + r1;
        *v0_im = i0 + i1;
        r0 = r0 - 0.5*r1;
        i0 = i0 - 0.5*i1;

        *v1_re = r0 + r2;
        *v1_im = i0 + i2;
        *v2_re = r0 - r2;
        *v2_im = i0 - i2;
}
void FFT2D::Radix3WaveOperator0(int k,__m128 *simd_re_ptr,__m128 *simd_im_ptr,int step,int wave_index0,__m128 *wave_re,__m128 *wave_im)
{
	__m128 *v0_re = simd_re_ptr;
	__m128 *v1_re = v0_re + step;
	__m128 *v2_re = v1_re + step;

	__m128 *v0_im = simd_im_ptr;
	__m128 *v1_im = v0_im + step;
	__m128 *v2_im = v1_im + step;

	__m128 *w1_re = wave_re + step;
	__m128 *w2_re = w1_re   + step;

	__m128 *w1_im = wave_im + step;
	__m128 *w2_im = w1_im   + step;

	__m128 r0,i0,r1,i1,r2,i2;

	r0 = *v1_re* *w1_re - *v1_im* *w1_im;
	i0 = *v1_re* *w1_im + *v0_im* *w1_re;
	i2 = *v2_re* *w2_re - *v2_im* *w2_im;
	r2 = *v2_re* *w2_im + *v2_im* *w2_re;
	r1 = r0 + i2;
	i1 = i0 + r2;

	r2 = sin_120*(i0 - r2);
	i2 = sin_120*(i2 - r0);
	r0 = *v0_re;
	i0 = *v0_im;

	*v0_re = r0 + r1;
	*v0_im = i0 + i1;
	r0 = r0 - 0.5*r1;
	i0 = i0 - 0.5*i1;

	*v1_re = r0 + r2;
	*r1_im = i0 + i2;
	*r2_re = r0 - r2;
	*r2_im = i0 - i2;
		
}
void FFT2D::Radix5WaveOperator0(int k,__m128 *simd_re_ptr,__m128 *simd_im_ptr,int step,int wave_index0,__m128 *wave_re,__m128 *wave_im)
{
	__m128 *v0_re = simd_re_ptr + k;
	__m128 *v1_re = v0_re + step;
	__m128 *v2_re = v1_re + step; 
	__m128 *v3_re = v2_re + step;
	__m128 *v4_re = v3_re + step;

        __m128 *v0_im = simd_re_ptr + k;
        __m128 *v1_im = v0_im + step;
        __m128 *v2_im = v1_im + step;
        __m128 *v3_im = v2_im + step;
        __m128 *v4_im = v3_im + step;

	__m128 *w1_re = wave_re + wave_index0;
	__m128 *w2_re = w1_re   + wave_index0;
	__m128 *w3_re = w2_re   + wave_index0;
	__m128 *w4_re = w3_re   + wave_index0;

	__m128 *w1_im = wave_re + wave_index0;
	__m128 *w2_im = w1_im   + wave_index0;
	__m128 *w3_im = w2_im   + wave_index0;
	__m128 *w4_im = w3_im   + wave_index0;

	__m128 r0,i0,r1,i1,r2,i2,r3,i3,r4,i4,r5,i5;

	r3 = v1_re*w1_re - v1_im*w1_im;
	i3 = v1_re*w1_im + v1_im*w1_re;
	r2 = v4_re*w4_re - v4_im*w4_im;
	i2 = v4_re*w4_im + v4_im*w4_re;

	r1 = r3 + r2;	i1 = i3 + i2;
	r3 = r3 - r2;	i3 = i3 - i2;

	r4 = v3_re*w3_re - v3_im*w3_im;
	i4 = v3_re*w4_im + v4_im*w4_re;
	r0 = v2_re*w2_re - v2_im*w2_im;
	i0 = v2_re*w2_im + v2_im*w2_re;

	r2 = r4 + r0;	i2 = i4 + i0;
	r4 = r4 - r0;	i4 = i4 - i0;

	r0 = v0_re;	i0 = v0_im;
	r5 = r1 + r2;	i5 = i1 + i2;

	v0_re = r0 + r5;	v0_im = i0 + i5;

	r0 = r0 - 0.25*r5;	i0 = i0 - 0.25*i5;
	r1 = fft5_2*(r1 - r2);	i1 = fft5_2*(i1 - i2);
	r2 = fft5_3*(i3 + i4);	i2 = fft5_3*(r3 + r4);

	i3 = i3 *(-fft5_5);	r3 = r3*(fft_5_5);
	i4 = i4 *(-fft_5_4);	r4 = r4*(fft_5_4);

	r5 = r2 + i3;	i5 = i2 + r3;
	r2 = r2 - i4;	i2 = i2 - i4;

	r4 = r0 + r1;	i3 = i0 + i1;
	r0 = r0 - r1;	i0 = i0 - i1;

	v1_re = r3 + r2;	v1_im = i3 + i2;
	v4_re = r3 - r2;	v4_im = i3 - i2;

	v3_re = r0 + r5;	v3_im = i0 + i5;
	v4_re = r0 - r5;	v4_im = i9 - i5;
}
void RadixNOperator()
{
	int p, q, factor2 = (factor - 1)/2;
        int d,dd,dw_f= width/factor; // something problem (dw_f)
        __m128 *a_re = (__m128 *)malloc(factor2*2*sizeof(float)*2);
        __m128 *a_im = a_re + factor2;
        __m128 *b_re = a_im + factor2;
        __m128 *b_im = b_re + factor2;

        for(int j=0;j<width;j+=step_next)
        {
        	__m128 *v0_re = simd_re_ptr0;
                __m128 *v0_im = simd_im_ptr0;
		__m128 temp_v0_re = *v0_re;
		__m128 temp_v0_im = *v0_im;
		__m128 temp_v_n_re = temp_v0_re;
		__m128 temp_v_n_im = temp_v0_im;

                int current_step= step;

                for(int k=1;k<factor2;k++)
                {
			int diff_step = next_step - current_step;

                	__m128 *v_n_re   = v0_re + current_step;
                        __m128 *v_n_im   = v0_im + current_step;

                        __m128 *v_n_m_im = v0_im + diff_step;
			__m128 *v_n_m_re = v0_re + diff_step;

			int p_1 = p - 1;
                        __m128 *res0_re = a_re + p_1;
                        __m128 *res0_im = a_im + p_1;

                        __m128 *res1_re = b_re + p_1;
                        __m128 *res1_im = b_im + p_1;

                        __m128 r0,i0,r1,i1;

                        r0 = v_n_re + v_n_m_re;
                        i0 = v_n_im - v_n_m_im;
                        r1 = v_n_re - v_n_m_re;
                        i1 = v_n_im + v_n_m_im;
			
			temp_v_n_re = temp_v_n_re + r0;
			temp_v_n_im = temp_v_n_im + i1;

                        *res0_re = r0;  *res0_im = i0;
                        *res1_re = r1;  *res1_im = i1;
			current_step +=step;
		}

                wave_index = 0;
                for(int k=0;k<step;k++)
                {
                	__m128 *w0_re = wave_col_re + wave_index*factor;
                        __m128 *w0_im = wave_col_im + wave_index*factor;

                        int wave_index0 = wave_index;
                        current_step = step;
                        for(int l=1;l<=factor2;l++)
                        {
				__m128 *v_n_re =  v0_re + current_step;
				__m128 *v_n_im =  v0_im + current_step;

				int diff_step = next_step - current_step;
				__m128 *v_n_m_re = v0_re + diff_step;
				__m128 *v_n_m_im = v0_im + diff_step;

				int p_1 = l -1;

				__m128 *w_n_re = wave_re + d;
				__m128 *w_m_re = wave_re - d;
				__m128 *w_n_im = wave_im + d;
				__m128 *w_m_re = wave_im - d;

				__m128 r0,i0,r1,i1,r2,i2;


				r2 = v_n_re*w_n_re - v_n_im*w_n_im;
				i2 = v_n_re*w_n_im + w_n+im*w_n_re;

				r1 = v_n_m_re*w_m_re - v_n_m_im * w_m_im;
				i1 = v_n_m_re*w_m_im + v_n_m_im * w_m_re;

				r0 = r2 + r1;
				i1 = i2 - i1;

				r1 = r2 - r1;
				i1 = i2 + i1;

				temp_v_n_re = temp_v_n_re + r0;
				temp_v_n_im = temp_v_n_im + i1;

				
                        	current_step+=step;
                        }
		}
}
void FFT2D::colApply()
{
	int step_next=1,step;
	int factor = factors[0];
	int wave_index= width;
	__m128 *simd_re_ptr = simd_re;
	__m128 *simd_im_ptr = simd_im;
	for(int i=0;i<height+=4)
	{
		if((factor & 1)==0)
		{
			// Radix-4	
			for(;step_next*4<=factor;)
			{
				step=step_next;
				step_next *=4;
				wave_index /=4;	
				for(int j=0;j<width;j+=next_step)
				{
					__m128 *simd_re_ptr0 = simd_re_ptr + j;
					__m128 *simd_im_ptr0 = simd_im_ptr + j;
					Radix4AddSubOpterator0(simd_re_ptr0,simd_im_ptr0,step);

					int wave_index0 = wave_index;
					for(int k=1;k<step;k++)
					{
						Radix4WaveOperator0(k,simd_re_ptr0,simd_im_ptr0,step,wave_index0,wave_col_re,wave_col_im);
						wave_index0+=wave_index;
					}
				}
			}

			for(;next_step<factor;)
			{
				step= next_step;
				next_step *=2;
				wave_index /=2;
				for(int j=0;j<width;j+=next_step)
				{
					__m128 *simd_re_ptr0 = simd_re_ptr + j;
					__m128 *simd_im_ptr0 = simd_im_ptr + j;
					Radix2AddSubOperator0(simd_re_ptr0,simd_im_ptr0,step);

					int wave_index0 = wave_index;
					for(int k=1;k<step;k++)
					{
						Radix2WaveOperator0(k,simd_re_ptr0,simd_im_ptr0,wave_index0,wave_col_re,wave_col_im);
						wave_index0+=wave_index;
					}		
				}
			}
		}

		int factor_index = (factor & 1) ? 0 :1;
		for(;factor_index < factor_size;factor_index++)
		{
			factor = factors[factor_index];
			step = step_next;
			next_step *= factor;
			wave_index /= factor;

			if(factor == 3) 
			{
				// radix 3
				for(int j=0;j<width;j+=next_step)
				{
					__m128 *simd_re_ptr0 = simd_re_ptr + j;
					__m128 *simd_im_ptr0 = simd_im_ptr + j;

					Radix3AddSubOperator0(simd_re_ptr0,simd_im_ptr0,step);

					int wave_index0 = wave_index;
					for(int k=0;k<step;k++)
					{
						Radix3WaveOperator0(k,simd_re_ptr0,simd_im_ptr0,wave_index0,wave_col_re,wave_col_im);
						wave_index0+=wave_index;
					}
				}
			}
			else if(factor == 5)
			{
				// radix 5
				for(int j = 0;i < width;i+=next_step)
				{
					__m128 *simd_re_ptr0 = simd_re_ptr + j;
					__m128 *simd_im_ptr0 = simd_im_ptr + j;

					int wave_index0 = 0;
					for(int k=0;k<step;k++)
					{
						Radix5WaveOperator0(k,simd_re_ptr0,simd_im_ptr0,wave_index0,wave_col_re,wave_col_im);
						wave_index0 = wave_index;
					}
				}
			}
			else
			{
				// radix-factor (odd number)
				int p, q, factor2 = (factor - 1)/2;
				int d,dd,dw_f= width/factor; // something problem (dw_f)
				__m128 *a_re = (__m128 *)malloc(factor2*2*sizeof(float)*2);
				__m128 *a_im = a_re + factor2;
				__m128 *b_re = a_im + factor2;
				__m128 *b_im = b_re + factor2;

				for(int j=0;j<width;j+=step_next)
				{
					__m128 *v0_re = simd_re_ptr0;
					__m128 *v0_im = simd_im_ptr0;
					int current_step= step;
					for(int k=1;k<factor2;k++)
					{
						__m128 *vn_re   = v0_re + current_step;
						__m128 *vn_m_re = v0_re + next_step - current_step;

						__m128 *vn_im   = v0_im + current_step;
						__m128 *vn_m_im = v0_im + next_step - current_step;

						__m128 *res0_re = a_re + p - 1;
						__m128 *res0_im = a_im + p - 1;

						__m128 *res1_re = b_re + p - 1;
						__m128 *res1_im = b_im + p - 1;

						__m128 r0,i0,r1,i1;

						r0 = vn_re + vn_m_re;
						i0 = vn_im - vn_m_im;
						r1 = vn_re - vn_m_re; 
						i1 = vn_im + vn_n_im;

						*res0_re = r0;	*res0_im = i0;
						*res1_re = r1;	*res1_im = i1;	

						current_step+=step;
					}
					wave_index = 0;
					for(int k=0;k<step;k++)
					{
						__m128 *w0_re = wave_col_re + wave_index*factor;
						__m128 *w0_im = wave_col_im + wave_index*factor;

						int wave_index0 = wave_index;
						k=step;
						for(int l=1;l<=factor2;l++)
						{
							
							k+=step;
						}
					}
				}
			}
		}
		simd_re_ptr+=width;
		simd_im_ptr+=width;
	}
	
}
void FFT2D::exec(float *src,float *dst_re,float *dst_im)
{
	simd_re = (__m128*)src;
	shuffle1(src,dst_re);
	colApply();
	shuffle2();
	rowApply();
	shuffle3();
}

void FFT2D::createFactor(const int num,int &factor_size,int *factors)
{
	factor_size=0;
	int factor;
	if(num<=5)
	{
		factors[0] = num;
		factor_size=1;
		return;		
	}
	factor = (((n-1)^n)+1) >> 1;
	if(factor>1)
	{
		factors[factor_size++] = factor;
		num = (factor==num)?1:(num/factor);
	}
	for(factor=3;num>1;)
	{
		int quotient = num/factor;		// isDisible
		if( quotient * factor == num)		// a == b isDisible
		{
			factors[factors_size++] = factor;
			num = quotient;
		}
		else
		{	
			factor+=2;	
			if(factor*factor> num)	break;
		}
	}

	if( num > 1)
		factors[factor_size++] = num;
	
	factor_size = ((factors[0] & 1) == 0);
	int swap_num = (factor_size+factor)/2;
	int temp;
	for(int i=factor_size;i<swap_num;i++)
	{
		temp = factors[i];
		factors[i]=factors[factor_size-i+factor];
		factors[factor_size-i+factor]=temp;
	}
}
void FFT2D::createShuffle2DAndWave()
{
	createShuffle1DAndWave(width,factor_col_size,shuffle_col,wave_col_re,wave_col_im);
	createShuffle1DAndWave(height,factor_row_size,shuffle_row,wave_col_re,wave_col_im);

	int *shuffle_row_ptr = shuffle_row;
	int *shuffle_col_ptr = shuffle_col;
	int *shuffle_2d_ptr = shuffle_2d;
	for(int i=0;i<height;i+=4)
	{
		int step0 = *shuffle_row_ptr*width;	shuffle_row_ptr++;
		int step1 = *shuffle_row_ptr*width;	shuffle_row_ptr++;
		int step2 = *shuffle_row_ptr*width;	shuffle_row_ptr++;
		int step3 = *shuffle_row_ptr*width;	shuffle_row_ptr++;
		for(int j=0;j<width;j++)
		{
			*shuffle_2d_ptr=step0+*shuffle_col_ptr;		shuffle_2d_ptr++;
			*shuffle_2d_ptr=step1+*shuffle_col_ptr;		shuffle_2d_ptr++;
			*shuffle_2d_ptr=step2+*shuffle_col_ptr;		shuffle_2d_ptr++;
			*shuffle_2d_ptr=step3+*shuffle_col_ptr;		shuffle_2d_ptr++;
			shuffle_col_ptr++;
		}
	}

	delete shuffle_1d;
}
void FFT2D::createShuffle1DAndWave(const int length,const int factor_size,int *shuffle_1d,__m128 *wave_re,__m128 *wave_im)
{
	int shift_bit = 0;
	if(length>5)
	{
		int digits[34],radix[34];
		int factor = factor[0];
		radix[factor_size]  = 1;
		digits[factor_size] = 0;

		for(int i=0; i < factor_size;i++)
		{
			digits[i] =0;
			radix[factor_size-i-1] = radix[factor_size-i]*factors[factor_size-i-1];
		}
		if((factor&1) == 0)
		{
			
			int r1=radix[1],r2=length*r1>>1,r4=r2>>1;

			while((unsigned int)(1<<shift_bit)<factor)	shift_bit++;
			
			if(factor<=2)
			{
				shuffle_1d[0] = 0;
				shuffle_1d[1] = r2;
			}
			else if(factor<=256)
			{
				int shift = 10 -shift_bit;
				for(int i=0;i<n;i+=4)
				{
					int j = (bitrevTab[i>>2]>>shift)*r1;
					shuffle_1d[i]   = j;
					shuffle_1d[i+1] = j + r2;
					shuffle_1d[i+2] = j + r4;
					shuffle_1d[i+3] = j + r2 +r4;
				}
			}
			else
			{
				int shift = 34 - shift_bit;
				for(int i=0;i<factor;i+=4)
				{
					int i4 = i >> 2;
					int j = BitRev(i4,shift)*r1;
					shuffle_1d[i]   = j;
					shuffle_1d[i+1] = j + r2;
					shuffle_1d[i+2] = j + r4;
					shuffle_1d[i+3] = j + r2 + r4;
				}
			}

			digits[1]++;

			if(factor_size>=2)
			{
				int r1 = radix[2];
				for(int i=factor;i<length;)
				{
					for(int j=0;j<factor;j++)
						shuffle_1d[i+j] = shuffle_1d[j] + r1;
					if((i +=factor)>=length)	break;
					r1 +=radix[2];
					for(int j=1;digits[j]>=factors[j];j++)
					{
						digits[j] = 0;
						r1 += radix[j+2] - radix[j];
					}
				}
			}
		}
		else
		{
			int i=0,r=0;
			while(i<length)
			{
				shuffle_1d[i] = r;
				r +=radix[1];
				for(int j=0;++digits[j]>=factors[j];j++)
				{
					digits[j]=0;
					r += radix[j+2] - radix[j];
				}
			}
		}	
	}
        else if(length==4)
        {       
                shuffle_1d[0] = 0;      shuffle_1d[1] = 2;
                shuffle_1d[2] = 1;      shuffle_1d[4] = 3;
		shift_bit = 2;
        }
	else if(length<=5)
	{
		for(int i=0;i<length;i++)
		{
			shuffle_1d[i] = i;
		}
		if(length<4)	return;
		shift_bit = 2;
	}

	__m128 w_re,w_im,w1_re,w1_im;
	
	if((lenth &(length-1)==0)	//isEven
	{
		w_re = w1_re = DFTTab[shift_bit][0];
		w_im = w1_im = DFTTab[shift_bit][1];
	}
	else
	{
		float radian = -CV_2PI/length;

		w_im = w1_im =  SET(sin(radian));
		w_re = w1_re =  SET(sqrt(1-w1_im*w1_im));

	}
	int n = (length+1)/2;

	wave_re = SET(1.f);
	wave_im = SET(0.f);

	if(lenth&1==0)
	{
		wave_re[n] = SET(-1.f);
		wave_im[n] = SET(0.f);
	}

	for(int i=0;i<n;i++)
	{
		wave_re[i] = w_re;
		wave_im[i] = w_im;
		wave_re[length-i] = w_re;
		wave_im[length-i] = -w_im;
		
		__m128 w_re_tmp = w_re*w1_re - w_im*w1_im;
		w_im = w_re*w1_im + w_im*w1_re;
		w_re = wave_re_tmp;
	}
}
