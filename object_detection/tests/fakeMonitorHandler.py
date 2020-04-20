#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# fake MonitorHandler imitating Visual Insights interface / displays metrics instead of plotting them

class FakeMonitorHandler():
    def __init__(self):
        pass

    def updateTrainMetrics(self, current_iter, max_iter, loss_cls, loss_bbox, epoch):
        print("[monitor_handler - train] iter = %d/%d (epoch %d) | loss_cls = = %f | loss_bbox = %f" % (current_iter, max_iter, epoch, loss_cls, loss_bbox))

    def updateTestMetrics(self, mAP):
        print("[monitor_handler - test] mAP = %f" % mAP)
