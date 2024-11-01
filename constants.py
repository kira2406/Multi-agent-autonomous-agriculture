class CropStages:
    NOT_PLANTED = -1
    GROWING = 0
    HARVEST = 1
    DRIED = 2
    DIED = 3

class CropTypes:
    EMPTY = 0
    WHEAT = 1
    RICE = 2
    CORN = 3

class CropGrowthRates:
    WHEAT = 0.04
    RICE = 0.025
    CORN = 0.02

class CropInfo:
    growth_rates = {
        CropTypes.WHEAT: 0.04,
        CropTypes.RICE: 0.025,
        CropTypes.CORN: 0.02
    }
    
    @staticmethod
    def get_growth_rate(crop_type):
        return CropInfo.growth_rates.get(crop_type, 0)
    
class GridElements:
    GRASS = 0
    START = 1
    PLOT = 2
    SEEDSTN1 = 3
    SEEDSTN2 = 4
    SEEDSTN3 = 5
    WATERTANK = 6
    SEEDERAGENT = 7
    WATERAGENT = 8
    HARVESTERAGENT = 9
    MARKET = 10
    GARBAGE = 11