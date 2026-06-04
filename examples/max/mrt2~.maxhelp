{
    "patcher": {
        "fileversion": 1,
        "appversion": {
            "major": 9,
            "minor": 1,
            "revision": 4,
            "architecture": "x64",
            "modernui": 1
        },
        "classnamespace": "box",
        "rect": [ 134.0, 126.0, 1375.0, 857.0 ],
        "default_fontname": "Menlo",
        "boxes": [
            {
                "box": {
                    "id": "obj-4",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 858.0, 733.0, 95.0, 22.0 ],
                    "saved_object_attributes": {
                        "parameter_enable": 0,
                        "parameter_mappable": 0
                    },
                    "text": "pattrstorage",
                    "varname": "u982001489"
                }
            },
            {
                "box": {
                    "id": "obj-1",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 4,
                    "outlettype": [ "", "", "", "" ],
                    "patching_rect": [ 858.0, 704.0, 74.0, 22.0 ],
                    "restore": {
                        "BufferSize": [ 2.0 ],
                        "Bypass": [ 0.0 ],
                        "Drumless": [ 0.0 ],
                        "MIDIGate": [ 0.0 ],
                        "Mute": [ 0.0 ],
                        "MuteDrums": [ 4.0 ],
                        "Notes": [ 2.4000000953674316 ],
                        "Reset": [ 0.0 ],
                        "Solo": [ 0.0 ],
                        "Style": [ 1.600000023841858 ],
                        "Temperature": [ 1.3 ],
                        "TopK": [ 40.0 ],
                        "Volume": [ 0.0 ],
                        "prompt0": [ "piano" ],
                        "prompt0weight": [ 0.5238343124860564 ],
                        "prompt1": [ "\"jazz drums\"" ],
                        "prompt1weight": [ 0.0 ],
                        "prompt2": [ "dubstep" ],
                        "prompt2weight": [ 0.0 ],
                        "prompt3": [ "\"synth pad\"" ],
                        "prompt3weight": [ 0.0 ]
                    },
                    "text": "autopattr",
                    "varname": "u479001342"
                }
            },
            {
                "box": {
                    "bubblesize": 14,
                    "id": "obj-5",
                    "maxclass": "preset",
                    "numinlets": 1,
                    "numoutlets": 5,
                    "outlettype": [ "preset", "int", "preset", "int", "" ],
                    "patching_rect": [ 858.0, 763.8297817707062, 100.33333333333334, 43.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 37.234042286872864, 267.0212746858597, 100.33333333333334, 43.0 ],
                    "preset_data": [
                        {
                            "number": 1,
                            "data": [ 5, "kslider", "kslider", "int", 36, 5, "midigate_toggle", "live.toggle", "float", 0.0, 5, "solo_tab", "live.tab", "float", 0.0, 5, "bufsize_menu", "live.menu", "float", 2.0, 5, "drumless_toggle", "live.toggle", "float", 0.0, 5, "bypass_toggle", "live.toggle", "float", 0.0, 5, "mute_toggle", "live.toggle", "float", 0.0, 5, "volume_slider", "live.slider", "float", 0.0, 5, "drums_dial", "live.dial", "float", 4.0, 5, "notes_dial", "live.dial", "float", 2.4000000953674316, 5, "style_dial", "live.dial", "float", 1.600000023841858, 5, "topk_dial", "live.dial", "float", 40.0, 5, "temperature_dial", "live.dial", "float", 1.2999999523162842, 5, "obj-10", "textedit", "restoretext", "piano", 5, "obj-14", "textedit", "restoretext", "\"jazz drums\"", 5, "obj-15", "textedit", "restoretext", "dubstep", 5, "obj-16", "textedit", "restoretext", "\"synth pad\"", 5, "obj-32", "slider", "float", 0.8571428656578064, 5, "obj-69", "slider", "float", 0.0, 5, "obj-70", "slider", "float", 0.0, 5, "obj-71", "slider", "float", 0.0 ]
                        },
                        {
                            "number": 2,
                            "data": [ 5, "kslider", "kslider", "int", 36, 5, "midigate_toggle", "live.toggle", "float", 0.0, 5, "solo_tab", "live.tab", "float", 0.0, 5, "bufsize_menu", "live.menu", "float", 2.0, 5, "drumless_toggle", "live.toggle", "float", 0.0, 5, "bypass_toggle", "live.toggle", "float", 0.0, 5, "mute_toggle", "live.toggle", "float", 0.0, 5, "volume_slider", "live.slider", "float", 0.0, 5, "drums_dial", "live.dial", "float", 4.0, 5, "notes_dial", "live.dial", "float", 2.4000000953674316, 5, "style_dial", "live.dial", "float", 1.600000023841858, 5, "topk_dial", "live.dial", "float", 40.0, 5, "temperature_dial", "live.dial", "float", 1.2999999523162842, 5, "obj-10", "textedit", "restoretext", "\"violin chamber ensemble\"", 5, "obj-14", "textedit", "restoretext", "\"jazz drums\"", 5, "obj-15", "textedit", "restoretext", "dubstep", 5, "obj-16", "textedit", "restoretext", "\"synth pad\"", 5, "obj-32", "slider", "float", 0.5913586616516113, 5, "obj-69", "slider", "float", 0.0, 5, "obj-70", "slider", "float", 0.0, 5, "obj-71", "slider", "float", 0.0 ]
                        },
                        {
                            "number": 3,
                            "data": [ 5, "kslider", "kslider", "int", 36, 5, "midigate_toggle", "live.toggle", "float", 0.0, 5, "solo_tab", "live.tab", "float", 0.0, 5, "bufsize_menu", "live.menu", "float", 2.0, 5, "drumless_toggle", "live.toggle", "float", 0.0, 5, "bypass_toggle", "live.toggle", "float", 0.0, 5, "mute_toggle", "live.toggle", "float", 0.0, 5, "volume_slider", "live.slider", "float", 0.0, 5, "drums_dial", "live.dial", "float", 4.0, 5, "notes_dial", "live.dial", "float", 2.4000000953674316, 5, "style_dial", "live.dial", "float", 1.600000023841858, 5, "topk_dial", "live.dial", "float", 40.0, 5, "temperature_dial", "live.dial", "float", 1.2999999523162842, 5, "obj-10", "textedit", "restoretext", "piano", 5, "obj-14", "textedit", "restoretext", "\"jazz drums\"", 5, "obj-15", "textedit", "restoretext", "dubstep", 5, "obj-16", "textedit", "restoretext", "\"synth pad\"", 5, "obj-32", "slider", "float", 0.2536625266075134, 5, "obj-69", "slider", "float", 0.0, 5, "obj-70", "slider", "float", 1.0, 5, "obj-71", "slider", "float", 0.516769528388977 ]
                        },
                        {
                            "number": 4,
                            "data": [ 5, "kslider", "kslider", "int", 36, 5, "midigate_toggle", "live.toggle", "float", 0.0, 5, "solo_tab", "live.tab", "float", 0.0, 5, "bufsize_menu", "live.menu", "float", 2.0, 5, "drumless_toggle", "live.toggle", "float", 0.0, 5, "bypass_toggle", "live.toggle", "float", 0.0, 5, "mute_toggle", "live.toggle", "float", 0.0, 5, "volume_slider", "live.slider", "float", 0.0, 5, "drums_dial", "live.dial", "float", 4.0, 5, "notes_dial", "live.dial", "float", 2.4000000953674316, 5, "style_dial", "live.dial", "float", 1.600000023841858, 5, "topk_dial", "live.dial", "float", 40.0, 5, "temperature_dial", "live.dial", "float", 1.2999999523162842, 5, "obj-10", "textedit", "restoretext", "piano", 5, "obj-14", "textedit", "restoretext", "\"jazz drums\"", 5, "obj-15", "textedit", "restoretext", "dubstep", 5, "obj-16", "textedit", "restoretext", "\"synth pad\"", 5, "obj-32", "slider", "float", 0.0, 5, "obj-69", "slider", "float", 0.6170893311500549, 5, "obj-70", "slider", "float", 0.0, 5, "obj-71", "slider", "float", 0.0 ]
                        }
                    ]
                }
            },
            {
                "box": {
                    "id": "obj-77",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 28.0, 355.0, 74.0, 22.0 ],
                    "text": "s to_mrt2"
                }
            },
            {
                "box": {
                    "id": "obj-76",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 28.0, 532.0, 74.0, 22.0 ],
                    "text": "s to_mrt2"
                }
            },
            {
                "box": {
                    "id": "obj-75",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 254.0, 355.0, 74.0, 22.0 ],
                    "text": "s to_mrt2"
                }
            },
            {
                "box": {
                    "id": "obj-74",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 254.0, 532.0, 74.0, 22.0 ],
                    "text": "s to_mrt2"
                }
            },
            {
                "box": {
                    "id": "obj-73",
                    "maxclass": "message",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 132.0, 739.0, 246.0, 22.0 ],
                    "text": "prompt 0 piano 0.523834"
                }
            },
            {
                "box": {
                    "floatoutput": 1,
                    "id": "obj-71",
                    "knobcolor": [ 0.6196078431372549, 0.9529411764705882, 0.6470588235294118, 1.0 ],
                    "maxclass": "slider",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 309.0, 461.0, 129.0, 21.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 196.0, 220.35106217861176, 129.0, 21.0 ],
                    "size": 1.0,
                    "varname": "prompt3weight"
                }
            },
            {
                "box": {
                    "floatoutput": 1,
                    "id": "obj-70",
                    "knobcolor": [ 0.9490196078431372, 0.9529411764705882, 0.6196078431372549, 1.0 ],
                    "maxclass": "slider",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 83.0, 463.0, 129.0, 21.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 196.0, 172.4787220954895, 129.0, 21.0 ],
                    "size": 1.0,
                    "varname": "prompt2weight"
                }
            },
            {
                "box": {
                    "floatoutput": 1,
                    "id": "obj-69",
                    "knobcolor": [ 0.9529411764705882, 0.6196078431372549, 0.6784313725490196, 1.0 ],
                    "maxclass": "slider",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 308.0, 285.0, 129.0, 21.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 196.0, 123.54255223274231, 129.0, 21.0 ],
                    "size": 1.0,
                    "varname": "prompt1weight"
                }
            },
            {
                "box": {
                    "id": "obj-66",
                    "maxclass": "newobj",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patcher": {
                        "fileversion": 1,
                        "appversion": {
                            "major": 9,
                            "minor": 1,
                            "revision": 4,
                            "architecture": "x64",
                            "modernui": 1
                        },
                        "classnamespace": "box",
                        "rect": [ 517.0, 266.0, 1000.0, 780.0 ],
                        "boxes": [
                            {
                                "box": {
                                    "id": "obj-37",
                                    "maxclass": "button",
                                    "numinlets": 1,
                                    "numoutlets": 1,
                                    "outlettype": [ "bang" ],
                                    "parameter_enable": 0,
                                    "patching_rect": [ 50.0, 163.0, 24.0, 24.0 ]
                                }
                            },
                            {
                                "box": {
                                    "id": "obj-33",
                                    "maxclass": "newobj",
                                    "numinlets": 4,
                                    "numoutlets": 1,
                                    "outlettype": [ "" ],
                                    "patching_rect": [ 50.0, 213.0, 121.0, 22.0 ],
                                    "text": "pack prompt 3 text 0."
                                }
                            },
                            {
                                "box": {
                                    "id": "obj-19",
                                    "maxclass": "newobj",
                                    "numinlets": 2,
                                    "numoutlets": 2,
                                    "outlettype": [ "", "" ],
                                    "patching_rect": [ 50.0, 90.0, 81.0, 22.0 ],
                                    "text": "route text"
                                }
                            },
                            {
                                "box": {
                                    "comment": "",
                                    "id": "obj-55",
                                    "index": 2,
                                    "maxclass": "inlet",
                                    "numinlets": 0,
                                    "numoutlets": 1,
                                    "outlettype": [ "" ],
                                    "patching_rect": [ 185.0, 45.0, 30.0, 30.0 ]
                                }
                            },
                            {
                                "box": {
                                    "comment": "",
                                    "id": "obj-56",
                                    "index": 1,
                                    "maxclass": "inlet",
                                    "numinlets": 0,
                                    "numoutlets": 1,
                                    "outlettype": [ "" ],
                                    "patching_rect": [ 50.0, 37.0, 30.0, 30.0 ]
                                }
                            },
                            {
                                "box": {
                                    "comment": "",
                                    "id": "obj-57",
                                    "index": 1,
                                    "maxclass": "outlet",
                                    "numinlets": 1,
                                    "numoutlets": 0,
                                    "patching_rect": [ 50.0, 295.0, 30.0, 30.0 ]
                                }
                            }
                        ],
                        "lines": [
                            {
                                "patchline": {
                                    "destination": [ "obj-33", 2 ],
                                    "order": 0,
                                    "source": [ "obj-19", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-37", 0 ],
                                    "order": 1,
                                    "source": [ "obj-19", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-57", 0 ],
                                    "source": [ "obj-33", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-33", 0 ],
                                    "source": [ "obj-37", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-33", 3 ],
                                    "order": 0,
                                    "source": [ "obj-55", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-37", 0 ],
                                    "order": 1,
                                    "source": [ "obj-55", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-19", 0 ],
                                    "source": [ "obj-56", 0 ]
                                }
                            }
                        ]
                    },
                    "patching_rect": [ 254.0, 497.0, 74.0, 22.0 ],
                    "text": "p prompt3"
                }
            },
            {
                "box": {
                    "id": "obj-65",
                    "maxclass": "newobj",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patcher": {
                        "fileversion": 1,
                        "appversion": {
                            "major": 9,
                            "minor": 1,
                            "revision": 4,
                            "architecture": "x64",
                            "modernui": 1
                        },
                        "classnamespace": "box",
                        "rect": [ 517.0, 266.0, 1000.0, 780.0 ],
                        "boxes": [
                            {
                                "box": {
                                    "id": "obj-37",
                                    "maxclass": "button",
                                    "numinlets": 1,
                                    "numoutlets": 1,
                                    "outlettype": [ "bang" ],
                                    "parameter_enable": 0,
                                    "patching_rect": [ 50.0, 163.0, 24.0, 24.0 ]
                                }
                            },
                            {
                                "box": {
                                    "id": "obj-33",
                                    "maxclass": "newobj",
                                    "numinlets": 4,
                                    "numoutlets": 1,
                                    "outlettype": [ "" ],
                                    "patching_rect": [ 50.0, 213.0, 121.0, 22.0 ],
                                    "text": "pack prompt 2 text 0."
                                }
                            },
                            {
                                "box": {
                                    "id": "obj-19",
                                    "maxclass": "newobj",
                                    "numinlets": 2,
                                    "numoutlets": 2,
                                    "outlettype": [ "", "" ],
                                    "patching_rect": [ 50.0, 90.0, 81.0, 22.0 ],
                                    "text": "route text"
                                }
                            },
                            {
                                "box": {
                                    "comment": "",
                                    "id": "obj-55",
                                    "index": 2,
                                    "maxclass": "inlet",
                                    "numinlets": 0,
                                    "numoutlets": 1,
                                    "outlettype": [ "" ],
                                    "patching_rect": [ 185.0, 45.0, 30.0, 30.0 ]
                                }
                            },
                            {
                                "box": {
                                    "comment": "",
                                    "id": "obj-56",
                                    "index": 1,
                                    "maxclass": "inlet",
                                    "numinlets": 0,
                                    "numoutlets": 1,
                                    "outlettype": [ "" ],
                                    "patching_rect": [ 50.0, 37.0, 30.0, 30.0 ]
                                }
                            },
                            {
                                "box": {
                                    "comment": "",
                                    "id": "obj-57",
                                    "index": 1,
                                    "maxclass": "outlet",
                                    "numinlets": 1,
                                    "numoutlets": 0,
                                    "patching_rect": [ 50.0, 295.0, 30.0, 30.0 ]
                                }
                            }
                        ],
                        "lines": [
                            {
                                "patchline": {
                                    "destination": [ "obj-33", 2 ],
                                    "order": 0,
                                    "source": [ "obj-19", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-37", 0 ],
                                    "order": 1,
                                    "source": [ "obj-19", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-57", 0 ],
                                    "source": [ "obj-33", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-33", 0 ],
                                    "source": [ "obj-37", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-33", 3 ],
                                    "order": 0,
                                    "source": [ "obj-55", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-37", 0 ],
                                    "order": 1,
                                    "source": [ "obj-55", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-19", 0 ],
                                    "source": [ "obj-56", 0 ]
                                }
                            }
                        ]
                    },
                    "patching_rect": [ 28.0, 495.0, 74.0, 22.0 ],
                    "text": "p prompt2"
                }
            },
            {
                "box": {
                    "id": "obj-64",
                    "maxclass": "newobj",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patcher": {
                        "fileversion": 1,
                        "appversion": {
                            "major": 9,
                            "minor": 1,
                            "revision": 4,
                            "architecture": "x64",
                            "modernui": 1
                        },
                        "classnamespace": "box",
                        "rect": [ 517.0, 266.0, 1000.0, 780.0 ],
                        "boxes": [
                            {
                                "box": {
                                    "id": "obj-37",
                                    "maxclass": "button",
                                    "numinlets": 1,
                                    "numoutlets": 1,
                                    "outlettype": [ "bang" ],
                                    "parameter_enable": 0,
                                    "patching_rect": [ 50.0, 163.0, 24.0, 24.0 ]
                                }
                            },
                            {
                                "box": {
                                    "id": "obj-33",
                                    "maxclass": "newobj",
                                    "numinlets": 4,
                                    "numoutlets": 1,
                                    "outlettype": [ "" ],
                                    "patching_rect": [ 50.0, 213.0, 121.0, 22.0 ],
                                    "text": "pack prompt 1 text 0."
                                }
                            },
                            {
                                "box": {
                                    "id": "obj-19",
                                    "maxclass": "newobj",
                                    "numinlets": 2,
                                    "numoutlets": 2,
                                    "outlettype": [ "", "" ],
                                    "patching_rect": [ 50.0, 90.0, 81.0, 22.0 ],
                                    "text": "route text"
                                }
                            },
                            {
                                "box": {
                                    "comment": "",
                                    "id": "obj-55",
                                    "index": 2,
                                    "maxclass": "inlet",
                                    "numinlets": 0,
                                    "numoutlets": 1,
                                    "outlettype": [ "" ],
                                    "patching_rect": [ 185.0, 45.0, 30.0, 30.0 ]
                                }
                            },
                            {
                                "box": {
                                    "comment": "",
                                    "id": "obj-56",
                                    "index": 1,
                                    "maxclass": "inlet",
                                    "numinlets": 0,
                                    "numoutlets": 1,
                                    "outlettype": [ "" ],
                                    "patching_rect": [ 50.0, 37.0, 30.0, 30.0 ]
                                }
                            },
                            {
                                "box": {
                                    "comment": "",
                                    "id": "obj-57",
                                    "index": 1,
                                    "maxclass": "outlet",
                                    "numinlets": 1,
                                    "numoutlets": 0,
                                    "patching_rect": [ 50.0, 295.0, 30.0, 30.0 ]
                                }
                            }
                        ],
                        "lines": [
                            {
                                "patchline": {
                                    "destination": [ "obj-33", 2 ],
                                    "order": 0,
                                    "source": [ "obj-19", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-37", 0 ],
                                    "order": 1,
                                    "source": [ "obj-19", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-57", 0 ],
                                    "source": [ "obj-33", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-33", 0 ],
                                    "source": [ "obj-37", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-33", 3 ],
                                    "order": 0,
                                    "source": [ "obj-55", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-37", 0 ],
                                    "order": 1,
                                    "source": [ "obj-55", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-19", 0 ],
                                    "source": [ "obj-56", 0 ]
                                }
                            }
                        ]
                    },
                    "patching_rect": [ 254.0, 315.0, 74.0, 22.0 ],
                    "text": "p prompt1"
                }
            },
            {
                "box": {
                    "id": "obj-58",
                    "maxclass": "newobj",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patcher": {
                        "fileversion": 1,
                        "appversion": {
                            "major": 9,
                            "minor": 1,
                            "revision": 4,
                            "architecture": "x64",
                            "modernui": 1
                        },
                        "classnamespace": "box",
                        "rect": [ 517.0, 276.0, 1000.0, 780.0 ],
                        "boxes": [
                            {
                                "box": {
                                    "id": "obj-37",
                                    "maxclass": "button",
                                    "numinlets": 1,
                                    "numoutlets": 1,
                                    "outlettype": [ "bang" ],
                                    "parameter_enable": 0,
                                    "patching_rect": [ 50.0, 163.0, 24.0, 24.0 ]
                                }
                            },
                            {
                                "box": {
                                    "id": "obj-33",
                                    "maxclass": "newobj",
                                    "numinlets": 4,
                                    "numoutlets": 1,
                                    "outlettype": [ "" ],
                                    "patching_rect": [ 50.0, 213.0, 160.0, 22.0 ],
                                    "text": "pack prompt 0 text 0."
                                }
                            },
                            {
                                "box": {
                                    "id": "obj-19",
                                    "maxclass": "newobj",
                                    "numinlets": 2,
                                    "numoutlets": 2,
                                    "outlettype": [ "", "" ],
                                    "patching_rect": [ 50.0, 90.0, 81.0, 22.0 ],
                                    "text": "route text"
                                }
                            },
                            {
                                "box": {
                                    "comment": "",
                                    "id": "obj-55",
                                    "index": 2,
                                    "maxclass": "inlet",
                                    "numinlets": 0,
                                    "numoutlets": 1,
                                    "outlettype": [ "" ],
                                    "patching_rect": [ 185.0, 45.0, 30.0, 30.0 ]
                                }
                            },
                            {
                                "box": {
                                    "comment": "",
                                    "id": "obj-56",
                                    "index": 1,
                                    "maxclass": "inlet",
                                    "numinlets": 0,
                                    "numoutlets": 1,
                                    "outlettype": [ "" ],
                                    "patching_rect": [ 50.0, 37.0, 30.0, 30.0 ]
                                }
                            },
                            {
                                "box": {
                                    "comment": "",
                                    "id": "obj-57",
                                    "index": 1,
                                    "maxclass": "outlet",
                                    "numinlets": 1,
                                    "numoutlets": 0,
                                    "patching_rect": [ 50.0, 295.0, 30.0, 30.0 ]
                                }
                            }
                        ],
                        "lines": [
                            {
                                "patchline": {
                                    "destination": [ "obj-33", 2 ],
                                    "order": 0,
                                    "source": [ "obj-19", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-37", 0 ],
                                    "order": 1,
                                    "source": [ "obj-19", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-57", 0 ],
                                    "source": [ "obj-33", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-33", 0 ],
                                    "source": [ "obj-37", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-33", 3 ],
                                    "order": 0,
                                    "source": [ "obj-55", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-37", 0 ],
                                    "order": 1,
                                    "source": [ "obj-55", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-19", 0 ],
                                    "source": [ "obj-56", 0 ]
                                }
                            }
                        ]
                    },
                    "patching_rect": [ 28.0, 315.0, 74.0, 22.0 ],
                    "text": "p prompt0"
                }
            },
            {
                "box": {
                    "floatoutput": 1,
                    "id": "obj-32",
                    "maxclass": "slider",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 83.0, 285.0, 129.0, 21.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 196.0, 78.86170148849487, 129.0, 21.0 ],
                    "size": 1.0,
                    "varname": "prompt0weight"
                }
            },
            {
                "box": {
                    "id": "obj-16",
                    "maxclass": "textedit",
                    "numinlets": 1,
                    "numoutlets": 4,
                    "outlettype": [ "", "int", "", "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 254.0, 423.0, 149.0, 31.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 37.234042286872864, 209.57446658611298, 148.93616914749146, 42.55319118499756 ],
                    "text": "\"synth pad\"",
                    "varname": "prompt3"
                }
            },
            {
                "box": {
                    "id": "obj-15",
                    "maxclass": "textedit",
                    "numinlets": 1,
                    "numoutlets": 4,
                    "outlettype": [ "", "int", "", "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 28.0, 423.0, 149.0, 31.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 37.234042286872864, 161.70212650299072, 148.93616914749146, 42.55319118499756 ],
                    "text": "dubstep",
                    "varname": "prompt2"
                }
            },
            {
                "box": {
                    "id": "obj-14",
                    "maxclass": "textedit",
                    "numinlets": 1,
                    "numoutlets": 4,
                    "outlettype": [ "", "int", "", "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 254.0, 244.0, 149.0, 31.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 37.234042286872864, 112.76595664024353, 148.93616914749146, 42.55319118499756 ],
                    "text": "\"jazz drums\"",
                    "varname": "prompt1"
                }
            },
            {
                "box": {
                    "id": "obj-10",
                    "keymode": 1,
                    "maxclass": "textedit",
                    "numinlets": 1,
                    "numoutlets": 4,
                    "outlettype": [ "", "int", "", "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 28.0, 244.0, 149.0, 31.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 37.234042286872864, 68.0851058959961, 148.93616914749146, 42.55319118499756 ],
                    "text": "piano",
                    "varname": "prompt0"
                }
            },
            {
                "box": {
                    "id": "obj-9",
                    "maxclass": "message",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 580.0, 733.0, 81.0, 22.0 ],
                    "text": "noteoff $1"
                }
            },
            {
                "box": {
                    "id": "obj-8",
                    "maxclass": "newobj",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 580.0, 704.0, 66.0, 22.0 ],
                    "text": "pipe 200"
                }
            },
            {
                "box": {
                    "id": "temperature_msg",
                    "maxclass": "message",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 480.0, 260.0, 110.0, 22.0 ],
                    "text": "temperature $1"
                }
            },
            {
                "box": {
                    "fontsize": 14.0,
                    "id": "title",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 20.0, 10.0, 1200.0, 23.0 ],
                    "text": "mrt2~ — Magenta RT v2 music model.  ⚠ Set Max audio SR to 48000 Hz (Options → Audio Status)."
                }
            },
            {
                "box": {
                    "id": "assets",
                    "maxclass": "message",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 28.0, 47.0, 380.00000566244125, 22.0 ],
                    "text": "assets ~/Documents/Magenta/magenta-rt-v2/resources"
                }
            },
            {
                "box": {
                    "id": "model",
                    "maxclass": "message",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 45.0, 84.0, 590.0, 22.0 ],
                    "text": "model ~/Documents/Magenta/magenta-rt-v2/models/mrt2_base/mrt2_base.mlxfn"
                }
            },
            {
                "box": {
                    "id": "s_init",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 28.0, 124.0, 74.0, 22.0 ],
                    "text": "s to_mrt2"
                }
            },
            {
                "box": {
                    "fontsize": 13.0,
                    "id": "ph",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 20.000000298023224, 184.8000027537346, 430.0, 22.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 37.0, 24.0, 312.0, 22.0 ],
                    "text": "── Prompts (prompt N \"text\" weight) ──"
                }
            },
            {
                "box": {
                    "fontsize": 13.0,
                    "id": "sh",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 480.0, 172.0, 170.0, 22.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 394.0, 21.0, 170.0, 22.0 ],
                    "text": "── Sampling ──"
                }
            },
            {
                "box": {
                    "id": "temperature_dial",
                    "maxclass": "live.dial",
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "", "float" ],
                    "parameter_enable": 1,
                    "patching_rect": [ 480.0, 200.0, 44.0, 48.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 394.0, 49.0, 44.0, 48.0 ],
                    "saved_attribute_attributes": {
                        "valueof": {
                            "parameter_initial": [ 1.3 ],
                            "parameter_initial_enable": 1,
                            "parameter_longname": "Temperature",
                            "parameter_mmax": 3.0,
                            "parameter_modmode": 3,
                            "parameter_shortname": "Temp",
                            "parameter_steps": 61,
                            "parameter_type": 0,
                            "parameter_unitstyle": 1
                        }
                    },
                    "varname": "Temperature"
                }
            },
            {
                "box": {
                    "id": "topk_dial",
                    "maxclass": "live.dial",
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "", "float" ],
                    "parameter_enable": 1,
                    "patching_rect": [ 540.0, 200.0, 44.0, 48.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 454.0, 49.0, 44.0, 48.0 ],
                    "saved_attribute_attributes": {
                        "valueof": {
                            "parameter_initial": [ 40 ],
                            "parameter_initial_enable": 1,
                            "parameter_longname": "TopK",
                            "parameter_mmax": 250.0,
                            "parameter_mmin": 1.0,
                            "parameter_modmode": 0,
                            "parameter_shortname": "TopK",
                            "parameter_type": 1,
                            "parameter_unitstyle": 0
                        }
                    },
                    "varname": "TopK"
                }
            },
            {
                "box": {
                    "fontsize": 13.0,
                    "id": "gh",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 660.0, 172.0, 300.0, 22.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 574.0, 21.0, 300.0, 22.0 ],
                    "text": "── Guidance (CFG) ──"
                }
            },
            {
                "box": {
                    "id": "style_dial",
                    "maxclass": "live.dial",
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "", "float" ],
                    "parameter_enable": 1,
                    "patching_rect": [ 660.0, 200.0, 44.0, 48.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 574.0, 49.0, 44.0, 48.0 ],
                    "saved_attribute_attributes": {
                        "valueof": {
                            "parameter_initial": [ 1.6 ],
                            "parameter_initial_enable": 1,
                            "parameter_longname": "Style",
                            "parameter_mmax": 7.0,
                            "parameter_mmin": -1.0,
                            "parameter_modmode": 3,
                            "parameter_shortname": "Style",
                            "parameter_type": 0,
                            "parameter_unitstyle": 1
                        }
                    },
                    "varname": "Style"
                }
            },
            {
                "box": {
                    "id": "style_msg",
                    "maxclass": "message",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 660.0, 260.0, 117.0, 22.0 ],
                    "text": "cfgmusiccoca $1"
                }
            },
            {
                "box": {
                    "id": "notes_dial",
                    "maxclass": "live.dial",
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "", "float" ],
                    "parameter_enable": 1,
                    "patching_rect": [ 720.0, 200.0, 44.0, 48.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 634.0, 49.0, 44.0, 48.0 ],
                    "saved_attribute_attributes": {
                        "valueof": {
                            "parameter_initial": [ 2.4 ],
                            "parameter_initial_enable": 1,
                            "parameter_longname": "Notes",
                            "parameter_mmax": 7.0,
                            "parameter_mmin": -1.0,
                            "parameter_modmode": 3,
                            "parameter_shortname": "Notes",
                            "parameter_type": 0,
                            "parameter_unitstyle": 1
                        }
                    },
                    "varname": "Notes"
                }
            },
            {
                "box": {
                    "id": "notes_msg",
                    "maxclass": "message",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 720.0, 284.0, 88.0, 22.0 ],
                    "text": "cfgnotes $1"
                }
            },
            {
                "box": {
                    "id": "drums_dial",
                    "maxclass": "live.dial",
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "", "float" ],
                    "parameter_enable": 1,
                    "patching_rect": [ 780.0, 200.0, 44.0, 48.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 694.0, 49.0, 44.0, 48.0 ],
                    "saved_attribute_attributes": {
                        "valueof": {
                            "parameter_initial": [ 4 ],
                            "parameter_initial_enable": 1,
                            "parameter_longname": "MuteDrums",
                            "parameter_mmax": 7.0,
                            "parameter_mmin": -1.0,
                            "parameter_modmode": 3,
                            "parameter_shortname": "Drums",
                            "parameter_type": 0,
                            "parameter_unitstyle": 1
                        }
                    },
                    "varname": "MuteDrums"
                }
            },
            {
                "box": {
                    "id": "drums_msg",
                    "maxclass": "message",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 780.0, 308.0, 88.0, 22.0 ],
                    "text": "cfgdrums $1"
                }
            },
            {
                "box": {
                    "id": "s_guidance",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 588.5, 349.60000520944595, 74.0, 22.0 ],
                    "text": "s to_mrt2"
                }
            },
            {
                "box": {
                    "fontsize": 13.0,
                    "id": "oh",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 900.0, 172.0, 350.0, 22.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 813.8297814130783, 21.0, 234.04255151748657, 22.0 ],
                    "text": "── Output & Config ──"
                }
            },
            {
                "box": {
                    "id": "volume_slider",
                    "maxclass": "live.slider",
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "", "float" ],
                    "parameter_enable": 1,
                    "patching_rect": [ 900.0, 200.0, 39.0, 95.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 814.0, 49.0, 39.0, 95.0 ],
                    "saved_attribute_attributes": {
                        "valueof": {
                            "parameter_initial": [ 0.0 ],
                            "parameter_initial_enable": 1,
                            "parameter_longname": "Volume",
                            "parameter_mmax": 0.0,
                            "parameter_mmin": -60.0,
                            "parameter_modmode": 3,
                            "parameter_shortname": "Vol",
                            "parameter_type": 0,
                            "parameter_unitstyle": 4
                        }
                    },
                    "varname": "Volume"
                }
            },
            {
                "box": {
                    "id": "volume_msg",
                    "maxclass": "message",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 900.0, 310.0, 80.0, 22.0 ],
                    "text": "volume $1"
                }
            },
            {
                "box": {
                    "id": "mute_toggle",
                    "maxclass": "live.toggle",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "parameter_enable": 1,
                    "patching_rect": [ 1019.0, 361.0, 21.0, 21.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 889.0, 98.0, 21.0, 21.0 ],
                    "saved_attribute_attributes": {
                        "valueof": {
                            "parameter_enum": [ "off", "on" ],
                            "parameter_longname": "Mute",
                            "parameter_mmax": 1,
                            "parameter_modmode": 0,
                            "parameter_shortname": "Mute",
                            "parameter_type": 2
                        }
                    },
                    "varname": "Mute"
                }
            },
            {
                "box": {
                    "fontsize": 11.0,
                    "id": "mute_label",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 1040.0, 361.0, 40.0, 19.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 910.0, 99.0, 40.0, 19.0 ],
                    "text": "Mute"
                }
            },
            {
                "box": {
                    "id": "mute_msg",
                    "maxclass": "message",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 1022.0, 386.0, 64.0, 22.0 ],
                    "text": "mute $1"
                }
            },
            {
                "box": {
                    "id": "bypass_toggle",
                    "maxclass": "live.toggle",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "parameter_enable": 1,
                    "patching_rect": [ 1091.0, 361.0, 21.0, 21.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 889.0, 66.0, 21.0, 21.0 ],
                    "saved_attribute_attributes": {
                        "valueof": {
                            "parameter_enum": [ "off", "on" ],
                            "parameter_longname": "Bypass",
                            "parameter_mmax": 1,
                            "parameter_modmode": 0,
                            "parameter_shortname": "Bypass",
                            "parameter_type": 2
                        }
                    },
                    "varname": "Bypass"
                }
            },
            {
                "box": {
                    "fontsize": 11.0,
                    "id": "bypass_label",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 1112.0000165700912, 361.1999996304512, 50.0, 19.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 910.2765955924988, 67.06382977962494, 50.0, 19.0 ],
                    "text": "Bypass"
                }
            },
            {
                "box": {
                    "id": "bypass_msg",
                    "maxclass": "message",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 1093.6000162959099, 386.0, 74.0, 22.0 ],
                    "text": "bypass $1"
                }
            },
            {
                "box": {
                    "id": "drumless_toggle",
                    "maxclass": "live.toggle",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "parameter_enable": 1,
                    "patching_rect": [ 1047.0, 215.0, 21.0, 21.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 961.0, 66.0, 21.0, 21.0 ],
                    "saved_attribute_attributes": {
                        "valueof": {
                            "parameter_enum": [ "off", "on" ],
                            "parameter_longname": "Drumless",
                            "parameter_mmax": 1,
                            "parameter_modmode": 0,
                            "parameter_shortname": "Drumless",
                            "parameter_type": 2
                        }
                    },
                    "varname": "Drumless"
                }
            },
            {
                "box": {
                    "fontsize": 11.0,
                    "id": "drumless_label",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 1068.0, 215.0, 61.0, 19.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 982.0, 67.0, 61.0, 19.0 ],
                    "text": "Drumless"
                }
            },
            {
                "box": {
                    "id": "drumless_msg",
                    "maxclass": "message",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 1050.0, 240.0, 88.0, 22.0 ],
                    "text": "drumless $1"
                }
            },
            {
                "box": {
                    "id": "reset_btn",
                    "maxclass": "live.button",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "parameter_enable": 1,
                    "patching_rect": [ 976.0, 219.0, 21.0, 21.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 37.234042286872864, 315.9574445486069, 28.723404049873352, 28.723404049873352 ],
                    "saved_attribute_attributes": {
                        "valueof": {
                            "parameter_enum": [ "off", "on" ],
                            "parameter_invisible": 1,
                            "parameter_longname": "Reset",
                            "parameter_mmax": 1.0,
                            "parameter_modmode": 0,
                            "parameter_shortname": "Reset",
                            "parameter_type": 4
                        }
                    },
                    "varname": "Reset"
                }
            },
            {
                "box": {
                    "fontsize": 11.0,
                    "id": "reset_label",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 997.4000002741814, 218.39999961853027, 45.0, 19.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 69.14893567562103, 321.27659344673157, 45.0, 19.0 ],
                    "text": "Reset"
                }
            },
            {
                "box": {
                    "id": "reset_msg",
                    "maxclass": "message",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 979.0, 244.0, 50.0, 22.0 ],
                    "text": "reset"
                }
            },
            {
                "box": {
                    "fontsize": 13.0,
                    "id": "bufh",
                    "linecount": 2,
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 1140.0, 200.0, 90.0, 37.0 ],
                    "presentation": 1,
                    "presentation_linecount": 2,
                    "presentation_rect": [ 890.2765955924988, 134.0425522327423, 90.0, 37.0 ],
                    "text": "Buffer\n(samples)"
                }
            },
            {
                "box": {
                    "fontsize": 12.0,
                    "id": "bufsize_menu",
                    "maxclass": "live.menu",
                    "numinlets": 1,
                    "numoutlets": 3,
                    "outlettype": [ "", "", "float" ],
                    "parameter_enable": 1,
                    "patching_rect": [ 1140.0, 244.0, 100.0, 18.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 890.2765955924988, 176.59574341773987, 100.0, 18.0 ],
                    "saved_attribute_attributes": {
                        "valueof": {
                            "parameter_enum": [ "2048", "4096", "8192" ],
                            "parameter_initial": [ 2 ],
                            "parameter_initial_enable": 1,
                            "parameter_longname": "BufferSize",
                            "parameter_mmax": 2,
                            "parameter_modmode": 0,
                            "parameter_shortname": "BufferSize",
                            "parameter_type": 2
                        }
                    },
                    "varname": "BufferSize"
                }
            },
            {
                "box": {
                    "id": "bufsize_msg",
                    "maxclass": "message",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 1140.0, 270.0, 102.0, 22.0 ],
                    "text": "buffersize $1"
                }
            },
            {
                "box": {
                    "id": "s_output",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 900.0000134110451, 418.40000623464584, 74.0, 22.0 ],
                    "text": "s to_mrt2"
                }
            },
            {
                "box": {
                    "fontsize": 13.0,
                    "id": "midih",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 497.0, 451.0, 370.0, 22.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 378.0, 192.0, 370.0, 22.0 ],
                    "text": "── MIDI Keyboard──────────────────────────────"
                }
            },
            {
                "box": {
                    "fontsize": 12.0,
                    "id": "solo_tab",
                    "maxclass": "live.tab",
                    "num_lines_patching": 1,
                    "num_lines_presentation": 1,
                    "numinlets": 1,
                    "numoutlets": 3,
                    "outlettype": [ "", "", "float" ],
                    "parameter_enable": 1,
                    "patching_rect": [ 498.0, 480.0, 100.0, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 379.0, 221.0, 100.0, 20.0 ],
                    "saved_attribute_attributes": {
                        "valueof": {
                            "parameter_enum": [ "Jam", "Solo" ],
                            "parameter_longname": "Solo",
                            "parameter_mmax": 1,
                            "parameter_modmode": 0,
                            "parameter_shortname": "Solo",
                            "parameter_type": 2,
                            "parameter_unitstyle": 9
                        }
                    },
                    "varname": "Solo"
                }
            },
            {
                "box": {
                    "id": "solo_scale",
                    "maxclass": "newobj",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "int" ],
                    "patching_rect": [ 498.0, 510.0, 45.0, 22.0 ],
                    "text": "* 127"
                }
            },
            {
                "box": {
                    "id": "solo_msg",
                    "maxclass": "message",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 498.0, 540.0, 110.0, 22.0 ],
                    "text": "unmaskwidth $1"
                }
            },
            {
                "box": {
                    "id": "midigate_toggle",
                    "maxclass": "live.toggle",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "parameter_enable": 1,
                    "patching_rect": [ 615.0, 481.0, 21.0, 21.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 496.0, 222.0, 21.0, 21.0 ],
                    "saved_attribute_attributes": {
                        "valueof": {
                            "parameter_enum": [ "off", "on" ],
                            "parameter_longname": "MIDIGate",
                            "parameter_mmax": 1,
                            "parameter_modmode": 0,
                            "parameter_shortname": "MIDIGate",
                            "parameter_type": 2
                        }
                    },
                    "varname": "MIDIGate"
                }
            },
            {
                "box": {
                    "fontsize": 11.0,
                    "id": "midigate_label",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 636.0, 481.0, 70.0, 19.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 517.0, 222.0, 70.0, 19.0 ],
                    "text": "MIDI Gate"
                }
            },
            {
                "box": {
                    "id": "midigate_msg",
                    "maxclass": "message",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 617.5, 540.0, 88.0, 22.0 ],
                    "text": "midigate $1"
                }
            },
            {
                "box": {
                    "id": "s_midi_ctrl",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 498.0, 572.0, 74.0, 22.0 ],
                    "text": "s to_mrt2"
                }
            },
            {
                "box": {
                    "id": "kslider",
                    "inputmode": 1,
                    "maxclass": "kslider",
                    "numinlets": 2,
                    "numoutlets": 2,
                    "outlettype": [ "int", "int" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 494.0, 606.0, 504.0, 76.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 378.0, 254.0, 504.0, 76.0 ]
                }
            },
            {
                "box": {
                    "id": "noteon_msg",
                    "maxclass": "message",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 494.0, 733.0, 80.0, 22.0 ],
                    "text": "noteon $1"
                }
            },
            {
                "box": {
                    "id": "s_midi_notes",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 494.0, 770.0, 74.0, 22.0 ],
                    "text": "s to_mrt2"
                }
            },
            {
                "box": {
                    "fontsize": 13.0,
                    "id": "engine_header",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 19.0, 670.0, 121.0, 22.0 ],
                    "text": "── Engine ───"
                }
            },
            {
                "box": {
                    "id": "r_mrt2",
                    "maxclass": "newobj",
                    "numinlets": 0,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 19.0, 700.0, 74.0, 22.0 ],
                    "text": "r to_mrt2"
                }
            },
            {
                "box": {
                    "id": "mrt",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "signal", "signal" ],
                    "patching_rect": [ 19.0, 739.0, 90.0, 22.0 ],
                    "text": "mrt2~"
                }
            },
            {
                "box": {
                    "id": "ezdac",
                    "maxclass": "ezdac~",
                    "numinlets": 2,
                    "numoutlets": 0,
                    "patching_rect": [ 19.0, 779.0, 45.0, 45.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 151.0, 266.0212746858597, 45.0, 45.0 ]
                }
            },
            {
                "box": {
                    "id": "topk_msg",
                    "maxclass": "message",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 540.0, 284.0, 70.0, 22.0 ],
                    "text": "topk $1"
                }
            }
        ],
        "lines": [
            {
                "patchline": {
                    "destination": [ "s_init", 0 ],
                    "source": [ "assets", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "bufsize_msg", 0 ],
                    "source": [ "bufsize_menu", 1 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "s_output", 0 ],
                    "source": [ "bufsize_msg", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "s_output", 0 ],
                    "source": [ "bypass_msg", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "bypass_msg", 0 ],
                    "source": [ "bypass_toggle", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "s_output", 0 ],
                    "source": [ "drumless_msg", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "drumless_msg", 0 ],
                    "source": [ "drumless_toggle", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "drums_msg", 0 ],
                    "source": [ "drums_dial", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "s_guidance", 0 ],
                    "source": [ "drums_msg", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "noteon_msg", 0 ],
                    "order": 1,
                    "source": [ "kslider", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-8", 0 ],
                    "order": 0,
                    "source": [ "kslider", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "s_midi_ctrl", 0 ],
                    "source": [ "midigate_msg", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "midigate_msg", 0 ],
                    "source": [ "midigate_toggle", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "s_init", 0 ],
                    "source": [ "model", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "ezdac", 1 ],
                    "source": [ "mrt", 1 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "ezdac", 0 ],
                    "source": [ "mrt", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "s_output", 0 ],
                    "source": [ "mute_msg", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "mute_msg", 0 ],
                    "source": [ "mute_toggle", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "s_midi_notes", 0 ],
                    "source": [ "noteon_msg", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "notes_msg", 0 ],
                    "source": [ "notes_dial", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "s_guidance", 0 ],
                    "source": [ "notes_msg", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-58", 0 ],
                    "source": [ "obj-10", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-64", 0 ],
                    "source": [ "obj-14", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-65", 0 ],
                    "source": [ "obj-15", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-66", 0 ],
                    "source": [ "obj-16", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-58", 1 ],
                    "source": [ "obj-32", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-77", 0 ],
                    "source": [ "obj-58", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-75", 0 ],
                    "source": [ "obj-64", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-76", 0 ],
                    "source": [ "obj-65", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-74", 0 ],
                    "source": [ "obj-66", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-64", 1 ],
                    "source": [ "obj-69", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-65", 1 ],
                    "source": [ "obj-70", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-66", 1 ],
                    "source": [ "obj-71", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-9", 0 ],
                    "source": [ "obj-8", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "s_midi_notes", 0 ],
                    "source": [ "obj-9", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "mrt", 0 ],
                    "order": 1,
                    "source": [ "r_mrt2", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-73", 1 ],
                    "order": 0,
                    "source": [ "r_mrt2", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "reset_msg", 0 ],
                    "source": [ "reset_btn", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "s_output", 0 ],
                    "source": [ "reset_msg", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "s_midi_ctrl", 0 ],
                    "source": [ "solo_msg", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "solo_msg", 0 ],
                    "source": [ "solo_scale", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "solo_scale", 0 ],
                    "source": [ "solo_tab", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "style_msg", 0 ],
                    "source": [ "style_dial", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "s_guidance", 0 ],
                    "source": [ "style_msg", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "temperature_msg", 0 ],
                    "source": [ "temperature_dial", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "s_guidance", 0 ],
                    "source": [ "temperature_msg", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "topk_msg", 0 ],
                    "source": [ "topk_dial", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "s_guidance", 0 ],
                    "source": [ "topk_msg", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "s_output", 0 ],
                    "source": [ "volume_msg", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "volume_msg", 0 ],
                    "source": [ "volume_slider", 0 ]
                }
            }
        ],
        "parameters": {
            "bufsize_menu": [ "BufferSize", "BufferSize", 0 ],
            "bypass_toggle": [ "Bypass", "Bypass", 0 ],
            "drumless_toggle": [ "Drumless", "Drumless", 0 ],
            "drums_dial": [ "MuteDrums", "Drums", 0 ],
            "midigate_toggle": [ "MIDIGate", "MIDIGate", 0 ],
            "mute_toggle": [ "Mute", "Mute", 0 ],
            "notes_dial": [ "Notes", "Notes", 0 ],
            "reset_btn": [ "Reset", "Reset", 0 ],
            "solo_tab": [ "Solo", "Solo", 0 ],
            "style_dial": [ "Style", "Style", 0 ],
            "temperature_dial": [ "Temperature", "Temp", 0 ],
            "topk_dial": [ "TopK", "TopK", 0 ],
            "volume_slider": [ "Volume", "Vol", 0 ],
            "parameterbanks": {
                "0": {
                    "index": 0,
                    "name": "",
                    "parameters": [ "-", "-", "-", "-", "-", "-", "-", "-" ],
                    "buttons": [ "-", "-", "-", "-", "-", "-", "-", "-" ]
                }
            },
            "inherited_shortname": 1
        },
        "autosave": 0,
        "oscreceiveudpport": 0
    }
}
