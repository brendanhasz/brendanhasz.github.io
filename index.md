Hi! I'm a PhD Student in Neuroscience at the 
[University of Minnesota](https://twin-cities.umn.edu/).
I research habitual and deliberative decision-making in the lab of 
[David Redish](http://redishlab.neuroscience.umn.edu/).
Right now I'm recording from neurons in the hippocampus and prefrontal cortex 
of rats as they run through mazes, and decoding what information is being 
represented in those two areas, and what information is flowing from each 
area to the other.
I'm also interested in data science and machine learning.

## Posts

{% for post in site.posts %}
  <strong> <a href="{{ post.url }}" title="{{ post.title }}">{{ post.title }}</a> </strong> <br />
  <span>{{ post.date | date_to_string }}</span> <br />
  {% if post.description %}
    {{ post.description }}
  {% endif %}
  <br />
{% endfor %}
