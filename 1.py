import numpy as np
import matplotlib.pyplot as plt

#초기 값
vi = 100
angle = np.deg2rad(45)
h = 100
g = 9.81

#공식
t = (vi*np.sin(angle)+np.sqrt((vi*np.sin(angle))**2+2*g*h)) / g
max_h = (vi * (np.sin(angle)))**2 / (2*g) + h
dis = vi*np.cos(angle)*t


#출력
print(max_h)
print(dis)
print

#배열 생성 및 그래프 작성
t_array = np.linspace(0,t,100)

dis_array = vi*np.cos(angle)*t_array
h_array = h + vi*np.sin(angle)*t_array - 0.5*g*t_array**2

h_max = np.max(h_array)
h_index = np.where(h_array == h_max)
h_max_dis = dis_array[h_index]
print("최고점 x: %f, 최고점 y: %f " %(h_max_dis, h_max))

print("끝점 x: %f , 끝점 y: %f " %(dis_array[-1], h_array[-1]))



plt.plot(dis_array, h_array)
plt.xlabel('distance(m)')
plt.ylabel('Height(m)')
plt.title('Graph')
plt.show()
    


