import matplotlib.pyplot as plt
Steps = [200, 300, 400, 500, 600,  800, 1000]

# Rotataion = {'CLDPS':{'PSNR':[16.42, 19.41, 21.8, 22.00, 22.46, 22.66, 22.74],
#                       'FID': [170.46, 92.25, 74.43, 63.425 ,52.46, 36.66, 33.66],
#                       'LPIPS':[0.521, 0.39242, 0.3205, 0.31232, 0.306, 0.3, 0.302]},
          
#           'BlindDPS':{'PSNR':[15.42, 15.06, 16.4, 16.5,  16.46, 16.36, 16.87],
#                       'FID': [350.163, 354.15, 374.2, 367., 368, 356, 343.76],
#                       'LPIPS':[0.601, 0.58, 0.554, 0.55, 0.56, 0.55, 0.552]},
#             'FastEM':{'PSNR':[15.24,  15.3, 15.44, 14.46, 15.64, 15.6, 15.96],
#                       'FID': [355.163, 355.14, 354.2, 325.5, 308, 276, 268.76],
#                       'LPIPS':[0.637, 0.599, 0.589, 0.610, 0.587, 0.599, 0.597]},           
#          'GibbsDDRM':{'PSNR':[16.8, 16.9, 17.41, 18.00,  18.46, 18.22, 18.46],
#                       'FID': [304.163, 303., 294, 288.2, 258, 246, 236.76],
#                       'LPIPS':[0.602, 0.612, 0.574, 0.59, 0.573, 0.569, 0.565]}}

# plt.plot(Steps, Rotataion['CLDPS']['PSNR'], label='CL-DPS')
# plt.plot(Steps, Rotataion['BlindDPS']['PSNR'], label='BlindDPS')
# plt.plot(Steps, Rotataion['FastEM']['PSNR'], label='FastEM')
# plt.plot(Steps, Rotataion['GibbsDDRM']['PSNR'], label='GibbsDDRM')
# plt.ylabel('PSNR')

# plt.plot(Steps, Rotataion['CLDPS']['FID'], label='CL-DPS')
# plt.plot(Steps, Rotataion['BlindDPS']['FID'], label='BlindDPS')
# plt.plot(Steps, Rotataion['FastEM']['FID'], label='FastEM')
# plt.plot(Steps, Rotataion['GibbsDDRM']['FID'], label='GibbsDDRM')
# plt.ylabel('FID')

# plt.plot(Steps, Rotataion['CLDPS']['LPIPS'], label='CL-DPS')
# plt.plot(Steps, Rotataion['BlindDPS']['LPIPS'], label='BlindDPS')
# plt.plot(Steps, Rotataion['FastEM']['LPIPS'], label='FastEM')
# plt.plot(Steps, Rotataion['GibbsDDRM']['LPIPS'], label='GibbsDDRM')
# plt.ylabel('LPIPS')

# plt.xlabel('Steps')

# plt.legend()
# plt.show()



Zoom = {'CLDPS':{'PSNR':[16.92, 18.41, 19.8, 20.40, 20.46, 20.66, 20.68],
                      'FID': [240.46, 142.25, 74.43, 63.425 ,52.46, 43.66, 42.61],
                      'LPIPS':[0.701, 0.5122, 0.46, 0.45232, 0.456, 0.435, 0.435]},
          'BlindDPS':{'PSNR':[15.42, 15.90, 16.1, 15.8, 15.9, 16.20, 16.39],
                      'FID': [350.163, 354.15, 374.2, 367., 368, 356, 292.91],
                      'LPIPS':[0.824, 0.824, 0.793, 0.785, 0.776, 0.785, 0.784]},
           'FastEM':{'PSNR':[16.54,  16.9, 17.04, 17.26, 17.64, 17.6, 17.68],
                      'FID': [355.163, 355.14, 354.2, 325.5, 308, 306, 303.25],
                      'LPIPS':[0.757, 0.719, 0.709, 0.690, 0.657, 0.632, 0.632]},           
           'GibbsDDRM':{'PSNR':[14.8, 14.9, 14.91, 15.00,  15.46, 15.22, 15.45],
                      'FID': [374.163, 363., 354, 348.2, 348, 336, 327.42],
                      'LPIPS':[0.902, 0.912, 0.94, 0.89, 0.873, 0.829, 0.802]}}

# plt.plot(Steps, Zoom['CLDPS']['PSNR'], label='CL-DPS')
# plt.plot(Steps, Zoom['BlindDPS']['PSNR'], label='BlindDPS')
# plt.plot(Steps, Zoom['FastEM']['PSNR'], label='FastEM')
# plt.plot(Steps, Zoom['GibbsDDRM']['PSNR'], label='GibbsDDRM')
# plt.ylabel('PSNR')

# plt.plot(Steps, Zoom['CLDPS']['FID'], label='CL-DPS')
# plt.plot(Steps, Zoom['BlindDPS']['FID'], label='BlindDPS')
# plt.plot(Steps, Zoom['FastEM']['FID'], label='FastEM')
# plt.plot(Steps, Zoom['GibbsDDRM']['FID'], label='GibbsDDRM')
# plt.ylabel('FID')

plt.plot(Steps, Zoom['CLDPS']['LPIPS'], label='CL-DPS')
plt.plot(Steps, Zoom['BlindDPS']['LPIPS'], label='BlindDPS')
plt.plot(Steps, Zoom['FastEM']['LPIPS'], label='FastEM')
plt.plot(Steps, Zoom['GibbsDDRM']['LPIPS'], label='GibbsDDRM')
plt.ylabel('LPIPS')

plt.xlabel('Steps')

plt.legend()
plt.show()



