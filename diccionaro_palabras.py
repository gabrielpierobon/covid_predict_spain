# import json

def diccionario_palabras(print_palabras=False):

    diccionaro_palabras = {

        'informacion_comunicaciones':{

            'A':['consultoria']

		},
		'comercio_transporte_hosteleria':{

			'A':['cereales', 'carne', 'fruta', 'verdura', 'bebidas', 'leche', 'tabaco', 'café',
               'mariscos', 'ropa', 'zapatos', 'zapatillas','electrodomesticos', 'perfume', 'farmacia',
               'muebles', 'lamparas', 'reloj', 'iphone', 'samsung', 'ordenador', 'PC', 'combustible',
               'ferreteria', 'fontaneria', 'panaderia', 'cerveza', 'vino', 'agua', 'cigarrillos',
               'television', 'pintura', 'alfombra', 'aspiradora', 'bicicleta', 'bicicleta estatica',
               'juguetes', 'cosméticos', 'camiseta'],

            'B':['hotel', 'booking', 'alojamiento', 'turismo', 'camping', 'restaurante', 'bar', 'catering']
		},
		'financieras_y_seguros':{

			'A':['obtener seguro', 'inversion', 'solicitar crédito','comprar acciones','solicitar hipoteca',
         		 'evaluacion de riesgos', 'tarjeta de credito']

		},

		'actividades_artisticas_recreativas_otrosservicios':{

			'A':['teatro', 'concierto', 'cine', 'biblioteca', 'casino','apuestas'],
			'B':['gimnasio', 'club deportivo', 'polideportivo', 'parque de atracciones'],
			'C':['lavanderia', 'peluqueria', 'centro de belleza']

		},	

		'admin_publica_educacion_sanidad':{

			'A':['administracion publica'],
			'B':['clases'],
			'C':['hospital', 'sanitarios']

		},

		'inmobiliarias':{

			'A': ['inmobiliaria', 'alquiler', 'contrato', 'idealista']

		},

		'industria':{

			'A':['industria']

		},

		'agricultura_ganadería_silvicultura_y_pesca':{

			'A':['agricultura', 'fruta', 'verdura', 'pescado']

		},

		'construccion':{

			'A':['vivienda', 'reparacion', 'instalacion', 'construccion', 'reforma']

		},

		'profesionales_cientificas_tecnicas_otras':{

			'A':['abogados', 'contadores', 'auditoria', 'fiscal', 'asesoria'],
			'B':['seguridad privada', 'investigacion', 'servicio limpieza', 'convencion', 'feria']

		},

		'impuestos':{

			'A':['impuestos']

		}
	}

    total_palabras = 0
    for i in list(diccionaro_palabras.keys()):
        for j in list(diccionaro_palabras[i].keys()):
            total_palabras = total_palabras+len(diccionaro_palabras[i][j])
    print(f'Diccionario cargado!\nCantidad de palabras en el diccionario: {total_palabras}\n')

    if print_palabras:
	    for i in list(diccionaro_palabras.keys()):
	        for j in list(diccionaro_palabras[i].keys()):
	            print(i+": Lista "+j+": \n"+str(diccionaro_palabras[i][j])+"\n")

    return diccionaro_palabras