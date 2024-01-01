#coding=gbk

# pid 调节实验
from IPython.display import display
from sympy import *
init_printing(use_latex='mathjax')

# Symbols p是微分算子 相当于s
Kp, Ki = symbols('K_p K_i')
p, R, L = symbols('p R L')

PI = Kp * (p + Ki) / p      # 串联式PI控制器
dc_motor = 1 / (R + p*L)     # 直流电机传函
G_open = PI * dc_motor
G_closed = G_open / (1 + G_open)

display(G_closed)
display(simplify(G_closed))
print(type(G_closed))

num, den = fraction(G_closed)
display('Numerator', num)
display('Denominator', simplify(den))

display(Poly(num, p).coeffs())
display(Poly(simplify(den), p).coeffs())

print(latex(Poly(num, p)))
print(latex(Poly(simplify(den), p)))

from pylab import *
import matplotlib
import matplotlib.pyplot as plt

import control
from control import TransferFunction, bode_plot
import numpy as np


mpl.rcParams['figure.dpi'] = 200
plt.style.use('ggplot')
matplotlib.use('TkAgg')

# 一阶系统bode图
numerator = [1]
denominator = [0.01/(2*np.pi), 1]
sys = TransferFunction(numerator, denominator)
frequency_range = 2*np.pi*np.logspace(-2, 4, 1000)
mag, phase, omega = bode_plot(sys, frequency_range, dB=True,  Hz=True, deg=True)


def get_coeffs_dc_motor_current_regulator(R, L, Bandwidth_Hz):
    Kp = Bandwidth_Hz * 2 * np.pi * L
    Ki = R / L
    return Kp, Ki


R = 0.1
L = 0.4

Kp, Ki = get_coeffs_dc_motor_current_regulator(R, L, 200)   # 带宽：200Hz

print('Kp = ', Kp, '|', 'Ki = ', Ki)

dc_motor = control.tf([1], [L, R])
pi_regulator = control.tf([Kp, Kp*Ki], [1, 0])
display(dc_motor)
display(pi_regulator)

open_sys = control.series(pi_regulator, dc_motor)   # 传函串联，相乘
closed_sys = control.feedback(open_sys, 1, sign=-1)  # 负反馈
display(open_sys)
display(closed_sys)
closed_sys = control.minreal(closed_sys)    # 最小实现，即零极点对消，二阶系统变成一阶系统
display(closed_sys)
print('Kp/L = ', Kp/L)
print('Kp/L/(2pi) = ', Kp/L/(2*pi))

# plot bode
plt.figure()
mag, phase, omega = bode_plot(closed_sys, 2*np.pi*np.logspace(-2,4,1000), dB=1, Hz=1, deg=1)

# 阶跃响应
T, yout = control.step_response(closed_sys, np.arange(0, 20, 1e-3))  # arange:输出类型array；range：输出类型list
plt.figure()
plt.plot(T, yout)
plt.ylim([0, 1.2])

# 上升时间
print(min(yout, key=lambda x : abs(x-0.9)))  # 使用lambda匿名函数，快速实现函数功能 lambda argument_list:expersion
print(T[(np.abs(yout - 0.9)).argmin()] * 1000, 'ms')

print('dc gain: ', control.dcgain(closed_sys))      # 零频增益（直流增益）
print('zero: ', control.zeros(closed_sys))
print('pole: ', control.poles(closed_sys))

# ===================================================== 电流环 =================================================
# -------------------------------------------------------------------------------------------------------------
# ===================================================== 转速环 =================================================
# from IPython.display import display
# from sympy import *
# init_printing(use_latex='mathjax')

p, L, Kp = symbols('p L K^i_p')
Gi_closed = 1 / (1 + L/Kp*p)    # 电流环传函，零极对消后的一阶系统

# Symbols
speedKp, speedKi = symbols('K^ω_p K^ω_i')
J_s, n_pp = symbols('J_s n_pp')

speedPI = speedKp*(p + speedKi)/p
dc_motor_motion = n_pp/J_s/p
G_open = dc_motor_motion * Gi_closed * speedPI
display('Open tf: ', G_open)
print(latex(G_open))

G_closed = G_open/(1 + G_open)
display('Closed tf: ', G_closed)
display(simplify(G_closed))
print(latex(simplify(G_closed)))

num, den = fraction(G_closed)
display('Numerator', num)
display('Denominator', simplify(den))


R = 0.1
L = 0.4
Kp, Ki = get_coeffs_dc_motor_current_regulator(R, L, 200)
Gi_closed = control.tf([1], [L/Kp, 1])
currentBandwidth_radPerSec = Kp/L
print('Bw_i:', currentBandwidth_radPerSec/(2*np.pi), 'Hz')

# 使用带宽的弧度计算Kp
def get_coeffs_dc_motor_SPEED_regurator(J_s, n_pp, delta, currentBandwidth_radPerSec):
    speedKi = currentBandwidth_radPerSec / delta**2
    speedKp = J_s/n_pp * delta * speedKi
    return speedKp, speedKi


n_pp = 2
J_s = 0.06
dc_motor_motion = control.tf([n_pp/J_s], [1, 0])
display('motion: ', dc_motor_motion)
delta = 4

fig1, fig2, fig3 = figure(1), figure(2), figure(3)

for delta in [1.5, 2, 4, 8, 10]:
    speedKp, speedKi = get_coeffs_dc_motor_SPEED_regurator(J_s, n_pp, delta, currentBandwidth_radPerSec)
    print(f'speedKp = {speedKp:g}', f'speedKi = {speedKi:g}', f'Wzero = {speedKi/2/np.pi} Hz', f'cutoff = {delta*speedKi/2/np.pi} Hz', f'ipole = {Kp/L/2/np.pi} Hz', sep='|')
    speedPI = control.tf([speedKp, speedKp*speedKi], [1, 0])
    Gw_open = dc_motor_motion*Gi_closed*speedPI
    Gw_closed = Gw_open / (1 + Gw_open)

    plt.figure(1)
    mag, phase, omega = bode_plot(Gw_open, 2 * np.pi * np.logspace(0, 4, 500), dB=1, Hz=1, deg=1, lw='0.5', label=f'{delta:g}')
    plt.figure(2)
    mag, phase, omega = bode_plot(Gw_closed, 2*np.pi*np.logspace(0, 4, 500), dB=1, Hz=1, deg=1, lw='0.5', label=f'{delta:g}')
    plt.figure(3)
    T, yout = control.step_response(Gw_closed, np.arange(0,0.05,1e-5))
    # 上升时间：
    print('\t', min(yout, key=lambda x : abs(x-0.9)), 'A')
    rise_time_ms = T[(np.abs(yout-0.9)).argmin()] * 1000
    print('\t', rise_time_ms, 'ms')
    plt.plot(T, yout, lw='0.5', label=f'{delta:g} | {rise_time_ms:g} ms')

fig2.axes[0].set_ylim([-3, 20]) # -3dB
fig3.axes[0].set_xlabel('time [s]')
fig3.axes[0].set_ylabel('speed [elec. deg]')

for i in [1, 2, 3]:
    plt.figure(i)
    plt.legend(loc='upper right')


delta = 5
desired_rise_time_ms = 2.3      # 阶跃上升时间

rise_time_ms = 100   # ms initial
Bw_current_Hz = 100   # Hz initial
while True:
    # 电流环 调整带宽, 改变上升时间
    if abs(rise_time_ms - desired_rise_time_ms) <= 0.1:
        break
    else:
        if rise_time_ms > desired_rise_time_ms:
            Bw_current_Hz += 10     # Hz    响应慢，带宽小，因此提高带宽
        else:
            Bw_current_Hz -= 10
            if Bw_current_Hz <= 0:
                raise Exception('Bandwidth is not less than 0Hz. Change bandwidth and try again.')  # 引发一个异常，程序中止运行
    print(f'Bw_current_Hz = {Bw_current_Hz}')
    R = 0.1
    L = 0.4
    Kp, Ki = get_coeffs_dc_motor_current_regulator(R, L, Bw_current_Hz)
    Gi_closed = control.tf([1], [L/Kp, 1])
    currentBandwidth_radPerSec = Kp/L

    # 速度环 speed loop
    n_pp = 2
    J_s = 0.06
    dc_motor_motion = control.tf([n_pp/J_s], [1, 0])
    speedKp, speedKi = get_coeffs_dc_motor_SPEED_regurator(J_s, n_pp, delta, currentBandwidth_radPerSec)
    print(f'speedKp = {speedKp:g}', f'speedKi = {speedKi:g}', f'wzero = {speedKi/2/np.pi} Hz', f'cutoff = {delta*speedKi/2/np.pi} Hz', f'ipole = {Kp/L/2/np.pi} Hz', sep=' | ')

    speedPI = control.tf([speedKp, speedKp*speedKi], [1, 0])
    Gw_open = dc_motor_motion * Gi_closed * speedPI
    Gw_closed = Gw_open / (1 + Gw_open)

    T, yout = control.step_response(Gw_closed, np.arange(0, 0.05, 1e-5))
    rise_time_ms = T[(np.abs(yout-0.9)).argmin()] * 1000     # <- Potential bug here! rise time is not always correct.

figure()
plt.plot(T, yout, lw='0.5', label=f'{delta:g} | {rise_time_ms:g} ms')
plt.xlabel('time [s]')
plt.ylabel('speed [elec. deg / sec]')
plt.legend(loc='upper right')

plt.show()


