def getAxons(I, HN, HM, O):
    axons_NI = [[1]*I for _ in range(HN)] # 2D
    axons_HMHN = [[[1]*HN for _ in range(HN)] for _ in range(HM)] # 3D
    axons_NO = [[1]*O for _ in range(HN)] # 2D
    axons_allLayers = [axons_NI,
                        *axons_HMHN, # unpack into HM axons_HN rows
                        axons_NO]
    return axons_allLayers