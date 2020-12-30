# da controllare
def serviceTimeCDF(s):
    if s < 0:
        Fs = 0
    elif s <= T*M**2/4:
        Fs = math.pi * s/(T*M**2)
    elif s <= T*M**2/2:
        Fs = math.pi * s/(T*M**2) - 4*s/(T*M**2)*math.acos(M /
                                                           2*math.sqrt(T/s)) + math.sqrt(4*s/(T*M**2)-1)
    else:
        Fs = 1.0
    return Fs

# 
def findQuantile(quantile, name, maxError):
    x = 0.0
    if name == 'serviceTime':
        error = quantile - serviceTimeCDF(x)
        while error > maxError:
            x += 0.1*error
            error = quantile - serviceTimeCDF(x)
    elif name == 'distance':
        error = quantile - distanceCDF(x)
        while error > maxError:
            x += 0.3*error
            error = quantile - distanceCDF(x)
    return x


'''
     find every quantile for both theoretical and sample distribution
'''

def fitDistribution(df, name, maxError):
    theoreticalQ = []
    sampleQ = []
    for i in range(1, len(df)):
        quantile = (i-0.5)/len(df)
        sq = df[name].quantile(quantile)
        tq = findQuantile(quantile, name, maxError)
        sampleQ.append(sq)
        theoreticalQ.append(tq)
        print(quantile, tq, sq)
    return [theoreticalQ, sampleQ]


'''
     draw a qq plot
'''

def qqPlot(theoreticalQ, sampleQ, name):
    slope, intercept, r_value, p_value, std_err = regr(theoreticalQ, sampleQ)

    plt.figure()
    plt.scatter(theoreticalQ, sampleQ, s=0.8, label=name, c='blue')
    y = [x*slope + intercept for x in theoreticalQ]
    plt.plot(theoreticalQ, y, 'r', label='Trend line')
    plt.text(0, max(sampleQ)*0.6, '\n\n$R^2$ = ' + str('%.6f' % r_value**2))
    if intercept > 0:
        plt.text(0, max(sampleQ)*0.55, 'y = ' + str('%.6f' %
                                                    slope) + 'x + ' + str('%.6f' % intercept))
    else:
        plt.text(0, max(sampleQ)*0.55, 'y = ' + str('%.6f' %
                                                    slope) + 'x ' + str('%.6f' % intercept))

    plt.xlabel('Theoretical Quantile')
    plt.ylabel('Sample Quantile')
    plt.title('QQ plot ' + name)
    plt.grid(True)
    plt.legend()
