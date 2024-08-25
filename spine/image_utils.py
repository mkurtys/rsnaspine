p = data_shape[i].reshape(-1)
m = data_image[i]

H,W = m.shape[:2]
s = 512/max(H,W)
    
p_512 = p*s 
m_512 = cv2.resize(m, dsize=None,fx=s,fy=s)
h,w = m_512.shape[:2]
m_512 = np.pad(m_512,[[0,512-h],[0,512-w], [0,0]],mode='constant',constant_values=0)


