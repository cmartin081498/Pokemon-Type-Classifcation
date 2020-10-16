#Curtis Martin
#E252H926
import pokebase as pb 
import json
#create a list of pokemon dictionarys
pokemon_list=[]
#key values for the pokemon dictionarys
pokemon_keys=['name','type','moves']
#create the dict with pokemon keys each value is None
pokemon_dict=dict.fromkeys(pokemon_keys,None)
#move dict
pokemon_moves_dict={}
#max number of pokemon
pokedex_max=808
#max number of moves
movedex_max=729
#create a dict of all pokemon moves
for i in range(1,movedex_max,1):
        pokemon_moves_dict[pb.move(i).name]=0
#fill out pokemon_dict
for i in range(1,pokedex_max,1):
    #i is the pokemon id that we want
    pokemon=pb.pokemon(i)
    #this is the name for the pokemon
    pokemon_dict['name']=pokemon.name
    pokemon_typelist=len(pokemon.types)
    pokemon_type=''
    #set the pokemons type
    for i in range(0,pokemon_typelist,1):
        pokemon_type=pokemon_type + pokemon.types[i].type.name
    pokemon_dict['type']=pokemon_type
    pokemon_dict['moves']=pokemon_moves_dict.copy()
    pokemon_moves=len(pokemon.moves)
    #check to see if the current pokemon can learn a move, if so set the value of the move to 1
    for i in range(0,pokemon_moves,1):
        move=pokemon.moves[i].move.name
        if move in pokemon_dict['moves'].keys():
            pokemon_dict['moves'][move]=1
    #append the pokemon to our poke list
    pokemon_list.append(pokemon_dict.copy())
filename='full_pokemon_dict.json'
with open(filename,'w')as f_obj:
    json.dump(pokemon_list,f_obj)