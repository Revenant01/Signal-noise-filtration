import numpy as np 
import matplotlib.pyplot as plt
import sounddevice as sd
#from scipy import pi
from scipy.fftpack import fft

t = np.linspace(0,9,18*1024)

x3 = np.array([0,0,0,0,0,246.93,0,0,220,0,130.81,164.81,220,246.93,0,164.81,207.6523,246.93,0])
x4 = np.array([329.63,311.27,329.63,311.127,329.63,0,293.66,261.63,0,0,0,0,0,0,0,0,0,0,261.63])
dauration = np.array([0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.5,0.25,0.25,0.25,0.25,0.5,0.25,0.25,0.25,0.25,0.5])
#start= np.array([0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.5,2.75,3,3.25,3.5,4,4.25,4.5,4.75,5,5.5])
# c3 = np.shape(x3)
# c4 = np.shape(x3)
# print(c3)
# print(c4)

x =0
start= 0.5

def u(t):
    return 1*(t>=0)

for i in range (0,19,1):
    F3 = x3[i]
    f4 = x4[i]
    #ti = start[i]
    Ti = 1.25*dauration[i]
    
    #x = x + (np.sin(2*np.pi*F3*t) + np.sin(2*np.pi*f4*t)) * ((t>=start)&(t<=start+Ti))
    x = x + (np.sin(2*np.pi*F3*t) + np.sin(2*np.pi*f4*t)) * (u(t-start)-u(t-start-Ti))
    start += Ti
print(x)
plt.plot(t,x)
plt.title('time domain Signal before adding the noise')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()  
sd.play(x,3*1024)

B = 1024
𝑁 = (18-0)*1024
t = np.linspace(0,3,N)
f = np. linspace(0 , 512 , int(𝑁/2))
magnitude = 1
[n1,n2] = np.random.randint(0, 512, 2)
fn1 = n1
fn2 = n2
noise1 = np.sin(2* (np.pi) * fn1 * t)
noise2 = np.sin(2* (np.pi) * fn2 * t)   
xn = x + noise1 + noise2

plt.plot (t, xn)
plt.title ('Time Domain Signal after adding the noise')
plt.xlabel ('Time')
plt.ylabel ('Amplitude')
plt.show ()

f1 = np. linspace(0 , 512 , int(𝑁/2))
x_f = fft(x)
x_f = 2/N * np.abs(x_f [0:np.int(N/2)])
plt.plot(f1, x_f)
plt.title('Frequency domain Signal before adding the noise')
plt.xlabel('Frequency in Hz')
plt.ylabel('Amplitude')
plt.show()  

f2 = np. linspace(0 , 512 , int(𝑁/2))
x_f2 = fft(xn)
x_f2 = 2/N * np.abs(x_f2 [0:np.int(N/2)])
plt.plot(f2, x_f2)
xfilter = xn -noise1 - noise2
plt.title('Frequency domain Signal after adding the noise')
plt.xlabel('Frequency in Hz')
plt.ylabel('Amplitude')
plt.show()  
    

maxAmp1 = max(x_f2) #The maximum amplitude of the noised signal array
tmp = 0
for i in range (len(x_f2)): # to find the index of the frequency of the maximum amplitude
    if(x_f2[i] == maxAmp1):
       maxAmp1 = i
       tmp = x_f2[i]
       x_f2[i] =0 # additional step to avoid catching the same maximum value
       break

maxAmp2 =max(x_f2) # to find the second maximum amplitude of the noised song
for k in range (len(x_f2)): # to find the index corresponding to the second maximum amplitude
     if (x_f2[k] == maxAmp2):
         maxAmp2 =k
         break
x_f2[maxAmp1] =tmp 
f1n = round(f1[maxAmp1])# to round the corrsponding frequency of the found index
f2n = round(f1[maxAmp2])     
#print(f1n , fn1 ,f2n ,fn2) #to make sure that the found frequencies match the noised generated.

xFilter= xn-(np.sin(2*f1n*np.pi*t)+np.sin(2*f2n*np.pi*t))

plt.plot (t, xfilter)
plt.title ('Time Domain Signal after removing the noise')
plt.xlabel ('Time')
plt.ylabel ('Amplitude')
plt.show ()


f = np. linspace(0 , B/2 , int(𝑁/2))
x_f3 = fft(xfilter)
x_f3 = 2/N * np.abs(x_f [0:np.int(N/2)])
plt.plot(f, x_f)
plt.title('Frequency domain Signal after removing the noise')
plt.xlabel('Frequency in Hz')
plt.ylabel('Amplitude')
plt.show() 
sd.play(xfilter,3*1024)
